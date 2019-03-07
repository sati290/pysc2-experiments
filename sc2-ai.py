import time
from os import path
from absl import app
import numpy as np
import gin
import gin.tf
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Conv2DTranspose, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from pysc2.lib import actions
from pysc2.lib.static_data import UNIT_TYPES
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, Features, FeatureType
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import SC2Env, Agent, Bot, Race, Difficulty


def preprocess_input(features, feature_spec):
    features = Lambda(lambda x: tf.split(x, x.get_shape()[1], axis=1))(features)

    for f in feature_spec:
        if f.type == FeatureType.CATEGORICAL:
            features[f.index] = Lambda(lambda x: tf.squeeze(x, axis=1))(features[f.index])
            features[f.index] = Lambda(lambda x: tf.cast(x, tf.int32))(features[f.index])
            features[f.index] = Lambda(lambda x: tf.one_hot(x, f.scale, axis=1))(features[f.index])
        else:
            features[f.index] = Lambda(lambda x: x / f.scale)(features[f.index])

    return features


def embedding_dims_for_feature(feature_spec):
    return np.maximum(np.int32(np.log(feature_spec.scale)), 1)


def embed_categorical(features, feature_spec):
    return [
        Conv2D(embedding_dims_for_feature(f), 1, data_format='channels_first')(features[f.index])
        if f.type == FeatureType.CATEGORICAL
        else features[f.index]
        for f in feature_spec
    ]


@gin.configurable
def output_block(state, target_shapes, feature_spec, activation='elu'):
    output = [None] * len(target_shapes)
    for shape, f in zip(target_shapes, feature_spec):
        if f.type == FeatureType.CATEGORICAL:
            embedded_shape = [embedding_dims_for_feature(f)] + shape[1:]
            output[f.index] = Dense(np.prod(embedded_shape), activation=activation)(state)
            output[f.index] = Reshape(embedded_shape)(output[f.index])
            output[f.index] = Conv2D(shape[0], 1, data_format='channels_first', activation='linear')(output[f.index])
        else:
            output[f.index] = Dense(np.prod(shape), activation='linear')(state)
            output[f.index] = Reshape(shape)(output[f.index])

    return output


@gin.configurable
def build_model(screen_features, minimap_features, dense_layer_size=512, activation='elu'):
    use_conv = False

    screen_output_shapes = [t.get_shape().as_list()[1:] for t in screen_features]
    minimap_output_shapes = [t.get_shape().as_list()[1:] for t in minimap_features]

    screen_features = embed_categorical(screen_features, SCREEN_FEATURES)
    screen_features = Concatenate(axis=1)(screen_features)

    minimap_features = embed_categorical(minimap_features, MINIMAP_FEATURES)
    minimap_features = Concatenate(axis=1)(minimap_features)

    # if use_conv:
    #     screen_conv = Conv2D(8, (1, 1), data_format='channels_first', name='screen_conv')(screen_input)
    #     minimap_conv = Conv2D(4, (1, 1), data_format='channels_first', name='minimap_conv')(minimap_input)
    #
    #     concat_inputs = Concatenate()([Flatten()(screen_conv), Flatten()(minimap_conv)])
    # else:
    concat_inputs = Concatenate()([Flatten()(screen_features), Flatten()(minimap_features)])

    dense = Dense(dense_layer_size, activation='elu')(concat_inputs)
    tf.summary.scalar('dense_zero_fraction', tf.nn.zero_fraction(dense))
    tf.summary.histogram('dense_output', dense)

    # if use_conv:
    #     screen_output = Dense(np.prod(screen_conv.shape[1:4]), activation='linear')(dense)
    #     screen_output = Reshape(screen_conv.shape[1:4], name='screen_output')(screen_output)
    #     screen_output = Conv2D(screen_shape[0], (1, 1), data_format='channels_first')(screen_output)
    #
    #     minimap_output = Dense(np.prod(minimap_conv.shape[1:4]), activation='linear')(dense)
    #     minimap_output = Reshape(minimap_conv.shape[1:4], name='minimap_output')(minimap_output)
    #     minimap_output = Conv2D(minimap_shape[0], (1, 1), data_format='channels_first')(minimap_output)
    # else:
    screen_output = output_block(dense, screen_output_shapes, SCREEN_FEATURES)
    minimap_output = output_block(dense, minimap_output_shapes, MINIMAP_FEATURES)

    return screen_output, minimap_output


def build_loss(inputs, outputs, feature_spec):
    losses = []
    for truth, prediction, spec in zip(inputs, outputs, feature_spec):
        if spec.type == FeatureType.CATEGORICAL:
            truth = tf.transpose(truth, (0, 2, 3, 1))
            prediction = tf.transpose(prediction, (0, 2, 3, 1))
            losses.append(tf.losses.softmax_cross_entropy(truth, prediction))
        else:
            losses.append(tf.losses.mean_squared_error(truth, prediction))

        tf.summary.scalar('loss_{}'.format(spec.name), losses[-1])

    return tf.reduce_mean(tf.stack(losses))


@gin.configurable
def main(args, learning_rate=0.0001):
    output_dir = path.join('runs', time.strftime('%Y%m%d-%H%M%S', time.localtime()))

    agent_interface_format = parse_agent_interface_format(feature_screen=16, feature_minimap=16)
    features = Features(agent_interface_format=agent_interface_format)

    action_spec = features.action_spec()
    obs_spec = features.observation_spec()

    screen_input = Input(shape=obs_spec['feature_screen'], name='screen_input')
    minimap_input = Input(shape=obs_spec['feature_minimap'], name='minimap_input')

    with tf.name_scope('preprocess'):
        screen_features = preprocess_input(screen_input, SCREEN_FEATURES)
        minimap_features = preprocess_input(minimap_input, MINIMAP_FEATURES)

    with tf.name_scope('model'):
        screen_output, minimap_output = build_model(screen_features, minimap_features)
        model = Model(inputs=[screen_input, minimap_input], outputs=screen_output + minimap_output)
        model.summary()

    with tf.name_scope('loss'):
        with tf.name_scope('screen'):
            screen_loss = build_loss(screen_features, screen_output, SCREEN_FEATURES)
            tf.summary.scalar('loss_mean', screen_loss)
        with tf.name_scope('minimap'):
            minimap_loss = build_loss(minimap_features, minimap_output, MINIMAP_FEATURES)
            tf.summary.scalar('loss_mean', minimap_loss)

        loss = tf.reduce_mean(tf.stack([screen_loss, minimap_loss]))

    global_step = tf.train.get_or_create_global_step()
    #learning_rate = tf.train.inverse_time_decay(0.001, global_step, 50000, 0.1)
    #learning_rate = tf.train.exponential_decay(0.0001, global_step, 50000, 0.1)
    opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss_total', loss)
    #merged_summaries = tf.summary.merge_all()

    env = SC2Env(map_name='Simple64', agent_interface_format=agent_interface_format, players=[
        Agent(Race.protoss),
        Bot(Race.protoss, Difficulty.easy)
    ], visualize=True)

    try:
        config_saver = gin.tf.GinConfigSaverHook(output_dir)
        with tf.train.MonitoredTrainingSession(hooks=[config_saver], checkpoint_dir=output_dir) as sess:
            while True:
                obs = env.reset()

                while True:
                    action = np.random.choice(obs[0].observation.available_actions)
                    args = [[np.random.randint(0, size) for size in arg.sizes] for arg in
                            action_spec.functions[action].args]

                    prev_obs = obs
                    obs = env.step([actions.FunctionCall(action, args)])

                    _, = sess.run((opt_op,), feed_dict={
                        screen_input: np.expand_dims(obs[0].observation['feature_screen'], 0),
                        minimap_input: np.expand_dims(obs[0].observation['feature_minimap'], 0)
                    })

                    if obs[0].step_type == StepType.LAST:
                        break

    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)
