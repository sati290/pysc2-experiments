import time
from os import path
from collections import namedtuple
import itertools
from absl import app, flags
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

FLAGS = flags.FLAGS

flags.DEFINE_boolean('save_checkpoints', False, '')
flags.DEFINE_boolean('visualize', False, '')
flags.DEFINE_boolean('profile', False, '')
flags.DEFINE_integer('step_limit', 0, '', lower_bound=0)
flags.DEFINE_string('config', 'config.gin', '')
flags.DEFINE_boolean('gpu_memory_allow_growth', False, '')
flags.DEFINE_float('gpu_memory_fraction', None, '', lower_bound=0, upper_bound=1)

EnvironmentSpec = namedtuple('EnvironmentSpec', ['action_spec', 'spaces'])

SpaceDesc = namedtuple('SpaceDesc', ['name', 'index', 'shape', 'features'])


def environment_spec(features):
    obs_spec = features.observation_spec()

    return EnvironmentSpec(features.action_spec(), [
        SpaceDesc('screen', 0, obs_spec['feature_screen'], SCREEN_FEATURES),
        SpaceDesc('minimap', 1, obs_spec['feature_minimap'], MINIMAP_FEATURES)
    ])


def preprocess_inputs(inputs, spaces):
    def one_hot_encode(x, scale):
        x = tf.squeeze(x, axis=1)
        x = tf.cast(x, tf.int32)
        return tf.one_hot(x, scale, axis=1)

    with tf.name_scope('preprocess_inputs'):
        outputs = [None] * len(spaces)
        for s in spaces:
            features = Lambda(lambda x: tf.split(x, x.get_shape()[1], axis=1))(inputs[s.index])

            for f in s.features:
                if f.type == FeatureType.CATEGORICAL:
                    features[f.index] = Lambda(lambda x: one_hot_encode(x, f.scale))(features[f.index])
                else:
                    features[f.index] = Lambda(lambda x: x / f.scale)(features[f.index])

            outputs[s.index] = features

    return outputs


@gin.configurable
def input_block(features, space_desc, conv_features=(8, 4), conv_activation='linear'):
    conv = Conv2D(conv_features[space_desc.index], 1, data_format='channels_first', activation=conv_activation,
                  name=space_desc.name + '_conv')

    features = Concatenate(axis=1, name=space_desc.name + '_concat_inputs')(features)
    features = conv(features)

    tf.summary.histogram('{}_conv_out'.format(space_desc.name), features)
    tf.summary.histogram('{}_conv_weights'.format(space_desc.name), conv.weights[0])

    return features


@gin.configurable
def output_block_spatial_one_conv(state, space_desc, spatial_features=(8, 4), spatial_activation='elu', output_activation='linear'):
    spatial_shape = (spatial_features[space_desc.index],) + space_desc.shape[1:]
    output_spatial = Dense(np.prod(spatial_shape), activation=spatial_activation)(state)
    output_spatial = Reshape(spatial_shape)(output_spatial)

    output_conv_features = np.sum([f.scale if f.type == FeatureType.CATEGORICAL else 1 for f in space_desc.features])
    output_conv = Conv2D(output_conv_features, 1, data_format='channels_first', activation=output_activation)(output_spatial)

    output = [None] * space_desc.shape[0]
    next_feature_index = 0
    for f in space_desc.features:
        name = '{}_{}_output'.format(space_desc.name, f.name)
        output_features = f.scale if f.type == FeatureType.CATEGORICAL else 1
        output[f.index] = Lambda(lambda x: x[:, next_feature_index:next_feature_index + output_features], name=name)(output_conv)
        next_feature_index = next_feature_index + output_features

    return output


@gin.configurable
def build_model(features, space_descs, dense_layer_size=(512,), activation='elu', output_block_fn=gin.REQUIRED):
    with tf.name_scope('model'):
        with tf.name_scope('input'):
            features = [input_block(features[s.index], s) for s in space_descs]

        with tf.name_scope('core'):
            features = [Flatten()(f) for f in features]
            features = Concatenate(name='concatenate_features')(features)

            dense = features
            for i, size in enumerate(dense_layer_size):
                op = Dense(size, activation=activation, name='state_dense_' + str(i))
                dense = op(dense)

                tf.summary.histogram('state_dense_' + str(i) + '_kernel_weights', op.weights[0])

            tf.summary.scalar('dense_zero_fraction', tf.nn.zero_fraction(dense))
            tf.summary.histogram('dense_input', features)
            tf.summary.histogram('dense_output', dense)

        with tf.name_scope('output'):
            outputs = [output_block_fn(dense, s) for s in space_descs]

    return outputs


def build_loss(inputs, outputs, feature_spec, palette):
    losses = []
    for truth, prediction, spec in zip(inputs, outputs, feature_spec):
        if spec.type == FeatureType.CATEGORICAL:
            truth = tf.transpose(truth, (0, 2, 3, 1))
            prediction = tf.transpose(prediction, (0, 2, 3, 1))
            losses.append(tf.losses.softmax_cross_entropy(truth, prediction))

            summary_image = tf.argmax(tf.concat([truth, prediction], 2), 3)
            summary_image = tf.gather(palette[spec.index], summary_image)
            tf.summary.image(spec.name, summary_image)
        else:
            losses.append(tf.losses.mean_squared_error(truth, prediction))

            summary_image = tf.concat([truth, prediction], 3)
            tf.summary.image(spec.name, tf.transpose(summary_image, (0, 2, 3, 1)))

        tf.summary.scalar(spec.name, losses[-1])

    return tf.reduce_mean(tf.stack(losses))


@gin.configurable
def main(args, learning_rate=0.0001, screen_size=16, minimap_size=16):
    output_dir = path.join('runs', time.strftime('%Y%m%d-%H%M%S', time.localtime()))

    gin.parse_config_file(FLAGS.config)
    gin.finalize()

    agent_interface_format = parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)
    env_spec = environment_spec(Features(agent_interface_format=agent_interface_format))

    inputs = [Input(shape=s.shape, name='{}_input'.format(s.name)) for s in env_spec.spaces]

    features = preprocess_inputs(inputs, env_spec.spaces)

    outputs = build_model(features, env_spec.spaces)
    model = Model(inputs=inputs, outputs=list(itertools.chain.from_iterable(outputs)))
    model.summary()

    feat_palettes = [[None] * len(s.features) for s in env_spec.spaces]
    for s in env_spec.spaces:
        for f in s.features:
            palette = f.palette
            if len(palette) < f.scale:
                palette = np.append(f.palette, [[255, 0, 255] * (f.scale - len(f.palette))], axis=0)
            feat_palettes[s.index][f.index] = tf.constant(palette, dtype=tf.uint8,
                                                          name='{}_{}_palette'.format(s.name, f.name))

    with tf.name_scope('loss'):
        losses = []
        for s in env_spec.spaces:
            with tf.name_scope(s.name):
                loss = build_loss(features[s.index], outputs[s.index], s.features, feat_palettes[s.index])
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
    ], visualize=FLAGS.visualize)

    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = FLAGS.gpu_memory_allow_growth
        if FLAGS.gpu_memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        hooks = [gin.tf.GinConfigSaverHook(output_dir)]
        if FLAGS.step_limit:
            hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.step_limit))
        if FLAGS.profile:
            hooks.append(tf.train.ProfilerHook(save_secs=60, output_dir=output_dir))
        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks, checkpoint_dir=output_dir,
                                               save_checkpoint_secs=3600 if FLAGS.save_checkpoints else None) as sess:
            while not sess.should_stop():
                obs = env.reset()

                while not sess.should_stop():
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
