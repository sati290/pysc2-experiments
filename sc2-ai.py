import time
from os import path
from collections import namedtuple
from absl import app, flags
import numpy as np
import gin
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Lambda, RNN, LSTMCell, Softmax
from tensorflow.python import debug as tf_debug
from pysc2.lib.features import parse_agent_interface_format, SCREEN_FEATURES, MINIMAP_FEATURES, Features, FeatureType
from pysc2.env.environment import StepType
from pysc2.env.sc2_env import SC2Env, Agent, Bot, Race, Difficulty
from pysc2.lib.actions import FunctionCall
from utils import print_parameter_summary, LogProgressHook

FLAGS = flags.FLAGS

flags.DEFINE_boolean('save_checkpoints', False, '')
flags.DEFINE_boolean('visualize', False, '')
flags.DEFINE_boolean('profile', False, '')
flags.DEFINE_integer('step_limit', None, '', lower_bound=0)
flags.DEFINE_string('config', 'config.gin', '')
flags.DEFINE_boolean('gpu_memory_allow_growth', False, '')
flags.DEFINE_float('gpu_memory_fraction', None, '', lower_bound=0, upper_bound=1)
flags.DEFINE_boolean('debug', False, '')

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
def predictions_output(state, space_desc, spatial_features=(8, 4), spatial_activation='elu',
                       output_activation='linear'):
    spatial_shape = (spatial_features[space_desc.index],) + space_desc.shape[1:]
    output_spatial = Dense(np.prod(spatial_shape), activation=spatial_activation)(state)
    output_spatial = Reshape(spatial_shape)(output_spatial)

    output_conv_features = np.sum([f.scale if f.type == FeatureType.CATEGORICAL else 1 for f in space_desc.features])
    output_conv = Conv2D(output_conv_features, 1, data_format='channels_first', activation=output_activation)(
        output_spatial)

    output = [None] * space_desc.shape[0]
    next_feature_index = 0
    for f in space_desc.features:
        name = '{}_{}_output'.format(space_desc.name, f.name)
        output_features = f.scale if f.type == FeatureType.CATEGORICAL else 1
        output[f.index] = Lambda(lambda x: x[:, next_feature_index:next_feature_index + output_features], name=name)(
            output_conv)
        next_feature_index = next_feature_index + output_features

    return output


@gin.configurable
def value_output(state, activation='linear'):
    out = Dense(1, activation=activation)(state)
    return tf.squeeze(out)


@gin.configurable
def policy_output(state, available_actions, action_spec):
    def probability_output(num_categories, name, mask=None):
        x = Dense(num_categories, activation='linear', name=name + '_logits')(state)
        if mask is not None:
            x = tf.where(mask > 0, x, -1000 * tf.ones_like(x), name=name + '_mask')
        return tfp.distributions.Categorical(logits=x, name=name + '_dist')

    fn_dist = probability_output(len(action_spec.functions), mask=available_actions, name='fn')

    arg_dists = {
        arg.name: probability_output(np.prod(arg.sizes), name='arg_{}'.format(arg.name))
        for arg in action_spec.types
    }

    return fn_dist, arg_dists


@gin.configurable
def sample_policy(policy):
    fn_dist, arg_dists = policy

    fn_sample = fn_dist.sample(name='fn_sample')
    arg_samples = {k: v.sample(name='arg_{}_sample'.format(k)) for k, v in arg_dists.items()}

    return fn_sample, arg_samples


@gin.configurable
def build_model(spatial_features, available_actions, env_spec, dense_layer_size=(512,), activation='elu',
                output_block_fn=gin.REQUIRED):
    with tf.name_scope('model'):
        with tf.name_scope('input'):
            spatial_features = [input_block(spatial_features[s.index], s) for s in env_spec.spaces]

        with tf.name_scope('core'):
            spatial_features = [Flatten()(f) for f in spatial_features]
            spatial_features = Concatenate(name='concatenate_features')(spatial_features)

            dense = spatial_features
            for i, size in enumerate(dense_layer_size):
                op = Dense(size, activation=activation, name='state_dense_' + str(i))
                dense = op(dense)

                tf.summary.histogram('state_dense_' + str(i) + '_kernel_weights', op.weights[0])

            tf.summary.scalar('dense_zero_fraction', tf.nn.zero_fraction(dense))
            tf.summary.histogram('dense_input', spatial_features)
            tf.summary.histogram('dense_output', dense)

        with tf.name_scope('prediction'):
            prediction = [output_block_fn(dense, s) for s in env_spec.spaces]

        with tf.name_scope('value'):
            value = value_output(dense)

        with tf.name_scope('policy'):
            policy = policy_output(dense, available_actions, env_spec.action_spec)

        with tf.name_scope('actions'):
            actions = sample_policy(policy)

    return prediction, value, policy, actions


def prediction_loss(truths, predictions, space_descs, palette):
    def spatial_loss(truth_features, predicted_features, space_desc):
        feature_losses = []
        for truth, prediction, spec in zip(truth_features, predicted_features, space_desc.features):
            if spec.type == FeatureType.CATEGORICAL:
                truth = tf.transpose(truth, (0, 2, 3, 1))
                prediction = tf.transpose(prediction, (0, 2, 3, 1))
                feature_losses.append(tf.losses.softmax_cross_entropy(truth, prediction))

                summary_image = tf.argmax(tf.concat([truth, prediction], 2), 3)
                summary_image = tf.gather(palette[space_desc.index][spec.index], summary_image)
                tf.summary.image(spec.name, summary_image)
            else:
                feature_losses.append(tf.losses.mean_squared_error(truth, prediction))

                summary_image = tf.concat([truth, prediction], 3)
                tf.summary.image(spec.name, tf.transpose(summary_image, (0, 2, 3, 1)))

            tf.summary.scalar(spec.name, feature_losses[-1])

        return tf.reduce_mean(tf.stack(feature_losses))

    with tf.name_scope('prediction_loss'):
        spatial_losses = []
        for s in space_descs:
            with tf.name_scope(s.name):
                loss = spatial_loss(truths[s.index], predictions[s.index], s)
                spatial_losses.append(loss)
                tf.summary.scalar('loss', loss)

        loss = tf.reduce_mean(tf.stack(spatial_losses))
        tf.summary.scalar('loss', loss)

    return loss


@gin.configurable
def value_loss(value_pred, returns, value_factor=1):
    with tf.name_scope('value_loss'):
        loss = tf.losses.mean_squared_error(returns, value_pred) * value_factor

        tf.summary.scalar('loss', loss)

    return loss


@gin.configurable
def policy_loss(action_dists, action_id, action_args, returns, value, policy_factor=0.1, entropy_factor=0.0001):
    with tf.name_scope('policy_loss'):
        def masked_arg_log_prob(arg_dist, arg_id, name):
            with tf.name_scope(name + 'log_prob'):
                clipped_arg_id = tf.maximum(arg_id, 0)
                log_prob = arg_dist.log_prob(clipped_arg_id)
                log_prob *= tf.cast(arg_id >= 0, tf.float32)
            return log_prob

        fn_dist, arg_dists = action_dists

        log_probs = [fn_dist.log_prob(action_id)] + [masked_arg_log_prob(v, action_args[k], k)
                                                     for k, v in arg_dists.items()]

        entropies = [fn_dist.entropy()] + [v.entropy() * tf.cast(action_args[k] >= 0, tf.float32)
                                           for k, v in arg_dists.items()]

        entropy = tf.reduce_mean(tf.reduce_sum(entropies))
        tf.summary.scalar('entropy', entropy)

        advantage = returns - value
        tf.summary.scalar('advantage', tf.reduce_mean(advantage))

        policy_loss = -tf.reduce_mean(tf.reduce_sum(log_probs) * tf.stop_gradient(advantage)) * policy_factor
        entropy_loss = -entropy * entropy_factor
        loss = policy_loss + entropy_loss
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('loss', loss)

    return loss


def actions_to_sc2(actions, action_spec):
    function = actions[0].item()
    args = [
        list(np.unravel_index(actions[1][arg.name].item(), arg.sizes))
        for arg in action_spec.functions[function].args
    ]

    return FunctionCall(function, args)


@gin.configurable
def main(args, learning_rate=0.0001, screen_size=16, minimap_size=16, discount=0.9):
    output_dir = path.join('runs', time.strftime('%Y%m%d-%H%M%S', time.localtime()))

    gin.parse_config_file(FLAGS.config)
    gin.finalize()

    agent_interface_format = parse_agent_interface_format(feature_screen=screen_size, feature_minimap=minimap_size)
    env_spec = environment_spec(Features(agent_interface_format=agent_interface_format))

    input_spaces = [Input(shape=s.shape, name='input_{}'.format(s.name)) for s in env_spec.spaces]
    input_available_actions = Input(shape=(len(env_spec.action_spec.functions),), name='input_available_actions')
    input_action_id = Input(shape=(), name='input_action_id')
    input_action_args = {arg.name: Input(shape=(), name='input_arg_{}_value'.format(arg.name)) for arg in
                         env_spec.action_spec.types}
    input_returns = Input(shape=(), name='input_returns')

    spatial_features = preprocess_inputs(input_spaces, env_spec.spaces)

    out_pred, out_value, out_policy, out_actions = build_model(spatial_features, input_available_actions, env_spec)
    print_parameter_summary()

    feat_palettes = [[None] * len(s.features) for s in env_spec.spaces]
    for s in env_spec.spaces:
        for f in s.features:
            palette = f.palette
            if len(palette) < f.scale:
                palette = np.append(f.palette, [[255, 0, 255] * (f.scale - len(f.palette))], axis=0)
            feat_palettes[s.index][f.index] = tf.constant(palette, dtype=tf.uint8,
                                                          name='{}_{}_palette'.format(s.name, f.name))

    loss_pred = prediction_loss(spatial_features, out_pred, env_spec.spaces, feat_palettes)
    loss = loss_pred + value_loss(out_value, input_returns) + policy_loss(out_policy, input_action_id,
                                                                          input_action_args, input_returns, out_value)

    global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.train.inverse_time_decay(0.001, global_step, 50000, 0.1)
    # learning_rate = tf.train.exponential_decay(0.0001, global_step, 50000, 0.1)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    no_op = tf.no_op()

    tf.summary.scalar('value', tf.reduce_mean(out_value))
    tf.summary.scalar('returns', tf.reduce_mean(input_returns))
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)

    env = SC2Env(map_name='Simple64', agent_interface_format=agent_interface_format, players=[
        Agent(Race.protoss),
        Bot(Race.protoss, Difficulty.very_easy)
    ], visualize=FLAGS.visualize)

    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = FLAGS.gpu_memory_allow_growth
        if FLAGS.gpu_memory_fraction:
            config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

        hooks = [gin.tf.GinConfigSaverHook(output_dir)]
        if FLAGS.step_limit:
            hooks.append(tf.train.StopAtStepHook(last_step=FLAGS.step_limit))
            hooks.append(LogProgressHook(FLAGS.step_limit))
        if FLAGS.profile:
            hooks.append(tf.train.ProfilerHook(save_secs=60, output_dir=output_dir))
        if FLAGS.debug:
            hooks.append(tf_debug.LocalCLIDebugHook())
        else:
            hooks.append(tf.train.NanTensorHook(loss))

        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks, checkpoint_dir=output_dir,
                                               save_checkpoint_secs=3600 if FLAGS.save_checkpoints else None) as sess:

            def feed_dict(obs, actions=None, returns=None):
                obs_features = [
                    np.expand_dims(obs[0].observation['feature_screen'], 0),
                    np.expand_dims(obs[0].observation['feature_minimap'], 0)
                ]

                obs_available_actions = np.zeros((1, len(env_spec.action_spec.functions)))
                obs_available_actions[:, obs[0].observation['available_actions']] = 1

                fd = dict(zip(input_spaces, obs_features))
                fd[input_available_actions] = obs_available_actions
                if actions is not None:
                    fd[input_action_id] = actions[0]
                    used_args = {
                        arg.name: actions[1][arg.name]
                        for arg in env_spec.action_spec.functions[actions[0].item()].args
                    }
                    for k, v in input_action_args.items():
                        if k in used_args:
                            fd[v] = used_args[k]
                        else:
                            fd[v] = np.reshape(-1, (1,))
                if returns is not None:
                    fd[input_returns] = np.reshape(returns, (1,))

                return fd

            while not sess.should_stop():
                obs = env.reset()

                while not sess.should_stop() and obs[0].step_type != StepType.LAST:
                    def step_fn(step_context):
                        actions = step_context.session.run(out_actions, feed_dict=feed_dict(obs))

                        next_obs = env.step([actions_to_sc2(actions, env_spec.action_spec)])

                        if next_obs[0].step_type == StepType.LAST:
                            returns = 0
                        else:
                            reward, next_value = step_context.session.run((loss_pred, out_value),
                                                                          feed_dict=feed_dict(next_obs))
                            returns = reward + discount * next_value

                        _ = step_context.run_with_hooks(train_op, feed_dict=feed_dict(obs, actions, returns))

                        return next_obs

                    obs = sess.run_step_fn(step_fn)

    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)
