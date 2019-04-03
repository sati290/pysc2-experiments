import time
from os import path
from absl import app, flags
import numpy as np
import gin
import gin.tf
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from pysc2.env.environment import StepType
from environment import SC2Environment
from agent import A2CAgent
from utils import print_parameter_summary, LogProgressHook

FLAGS = flags.FLAGS

flags.DEFINE_boolean('save_checkpoints', False, '')
flags.DEFINE_boolean('profile', False, '')
flags.DEFINE_integer('step_limit', None, '', lower_bound=0)
flags.DEFINE_string('config', 'config.gin', '')
flags.DEFINE_boolean('gpu_memory_allow_growth', False, '')
flags.DEFINE_float('gpu_memory_fraction', None, '', lower_bound=0, upper_bound=1)
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('trace', False, '')


@gin.configurable
def main(args):
    output_dir = path.join('runs', time.strftime('%Y%m%d-%H%M%S', time.localtime()))

    gin.parse_config_file(FLAGS.config)
    gin.finalize()

    with SC2Environment() as env:
        actor = A2CAgent(env.spec)

        print_parameter_summary()

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
            hooks.append(tf.train.NanTensorHook(actor.loss))
        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks, checkpoint_dir=output_dir,
                                               save_checkpoint_secs=3600 if FLAGS.save_checkpoints else None) as sess:

            summary_writer = tf.summary.FileWriterCache.get(output_dir)

            env.start()

            while not sess.should_stop():
                obs = env.reset()

                episode_sum_reward = 0
                while not sess.should_stop() and obs[0].step_type != StepType.LAST:
                    def step_fn(step_context):
                        action = actor.get_action(step_context, obs[0].observation)

                        next_obs = env.step([action])

                        actor.receive_reward(step_context, obs[0].observation, action, next_obs[0].reward,
                                             next_obs[0].observation, next_obs[0].step_type == StepType.LAST)

                        return next_obs

                    obs = sess.run_step_fn(step_fn)
                    episode_sum_reward += obs[0].reward

                global_step = sess.run_step_fn(lambda step_context: step_context.session.run(tf.train.get_global_step()))
                summary_writer.add_summary(
                    tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=episode_sum_reward)]),
                    global_step=global_step
                )


if __name__ == '__main__':
    app.run(main)
