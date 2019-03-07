import time
from os import path
from collections import namedtuple
import itertools
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


EnvironmentSpec = namedtuple('EnvironmentSpec', ['action_spec', 'spaces'])

SpaceDesc = namedtuple('SpaceDesc', ['name', 'index', 'shape', 'features'])


def environment_spec(features):
    obs_spec = features.observation_spec()

    return EnvironmentSpec(features.action_spec(), [
        SpaceDesc('screen', 0, obs_spec['feature_screen'], SCREEN_FEATURES),
        SpaceDesc('minimap', 1, obs_spec['feature_minimap'], MINIMAP_FEATURES)
    ])


def preprocess_inputs(inputs, spaces):
    with tf.name_scope('preprocess_inputs'):
        outputs = [None] * len(spaces)
        for s in spaces:
            features = Lambda(lambda x: tf.split(x, x.get_shape()[1], axis=1))(inputs[s.index])

            for f in s.features:
                if f.type == FeatureType.CATEGORICAL:
                    features[f.index] = Lambda(lambda x: tf.squeeze(x, axis=1))(features[f.index])
                    features[f.index] = Lambda(lambda x: tf.cast(x, tf.int32))(features[f.index])
                    features[f.index] = Lambda(lambda x: tf.one_hot(x, f.scale, axis=1))(features[f.index])
                else:
                    features[f.index] = Lambda(lambda x: x / f.scale)(features[f.index])

            outputs[s.index] = features

    return outputs


def embedding_dims_for_feature(feature_spec):
    return np.maximum(np.int32(np.log(feature_spec.scale)), 1)


def embed_categorical(features, space):
    return [
        Conv2D(embedding_dims_for_feature(f), 1, data_format='channels_first',
               name='{}_{}_conv_categorical'.format(space.name, f.name))(features[f.index])
        if f.type == FeatureType.CATEGORICAL
        else features[f.index]
        for f in space.features
    ]


@gin.configurable
def output_block(state, space_desc, activation='elu'):
    output = [None] * space_desc.shape[0]
    for f in space_desc.features:
        name = '{}_{}_output'.format(space_desc.name, f.name)
        with tf.name_scope(name):
            if f.type == FeatureType.CATEGORICAL:
                embedded_shape = (embedding_dims_for_feature(f),) + space_desc.shape[1:]
                output[f.index] = Dense(np.prod(embedded_shape), activation=activation)(state)
                output[f.index] = Reshape(embedded_shape)(output[f.index])
                output[f.index] = Conv2D(f.scale, 1, data_format='channels_first', activation='linear', name=name)(output[f.index])
            else:
                shape = (1,) + space_desc.shape[1:]
                output[f.index] = Dense(np.prod(shape), activation='linear')(state)
                output[f.index] = Reshape(shape, name=name)(output[f.index])

    return output


@gin.configurable
def build_model(features, space_descs, dense_layer_size=512, activation='elu'):
    with tf.name_scope('model'):
        with tf.name_scope('embed_categorical'):
            features = [embed_categorical(features[s.index], s) for s in space_descs]

        with tf.name_scope('core'):
            with tf.name_scope('concatenate_features'):
                features = [Concatenate(axis=1)(x) for x in features]
                features = [Flatten()(f) for f in features]
                features = Concatenate(name='concatenate_features')(features)

            dense = Dense(dense_layer_size, activation=activation)(features)

            tf.summary.scalar('dense_zero_fraction', tf.nn.zero_fraction(dense))
            tf.summary.histogram('dense_output', dense)

        with tf.name_scope('output'):
            outputs = [output_block(dense, s) for s in space_descs]


    return outputs


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
    env_spec = environment_spec(Features(agent_interface_format=agent_interface_format))

    inputs = [Input(shape=s.shape, name='{}_input'.format(s.name)) for s in env_spec.spaces]

    features = preprocess_inputs(inputs, env_spec.spaces)

    outputs = build_model(features, env_spec.spaces)
    model = Model(inputs=inputs, outputs=list(itertools.chain.from_iterable(outputs)))
    model.summary()

    with tf.name_scope('loss'):
        losses = []
        for s in env_spec.spaces:
            with tf.name_scope(s.name):
                loss = build_loss(features[s.index], outputs[s.index], s.features)
                losses.append(loss)
                tf.summary.scalar('loss', loss)

        loss = tf.reduce_mean(tf.stack(losses))

    global_step = tf.train.get_or_create_global_step()
    #learning_rate = tf.train.inverse_time_decay(0.001, global_step, 50000, 0.1)
    #learning_rate = tf.train.exponential_decay(0.0001, global_step, 50000, 0.1)
    opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss_total', loss)

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
                            env_spec.action_spec.functions[action].args]

                    obs = env.step([actions.FunctionCall(action, args)])

                    obs_features = [
                        np.expand_dims(obs[0].observation['feature_screen'], 0),
                        np.expand_dims(obs[0].observation['feature_minimap'], 0)
                    ]

                    _, = sess.run((opt_op,), feed_dict=dict(zip(inputs, obs_features)))

                    if obs[0].step_type == StepType.LAST:
                        break

    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)
