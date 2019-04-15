import time
from os import path
from collections import deque
from absl import app, flags
import numpy as np
import gin
import gin.tf
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from environment import SC2Environment
from agents import A2CAgent
from utils import print_parameter_summary, LogProgressHook

FLAGS = flags.FLAGS
flags.DEFINE_integer('save_checkpoint_secs', None, '')
flags.DEFINE_alias('s', 'save_checkpoint_secs')
flags.DEFINE_integer('save_checkpoint_steps', None, '')
flags.DEFINE_string('resume', None, '')
flags.DEFINE_alias('r', 'resume')
flags.DEFINE_integer('step_limit', None, '', lower_bound=0)
flags.DEFINE_alias('l', 'step_limit')
flags.DEFINE_string('run_name', None, '')
flags.DEFINE_alias('n', 'run_name')
flags.DEFINE_string('config', 'config.gin', '')
flags.DEFINE_alias('c', 'config')
flags.DEFINE_string('run_dir', 'runs', '')
flags.DEFINE_alias('d', 'run_dir')
flags.DEFINE_boolean('gpu_memory_allow_growth', False, '')
flags.DEFINE_float('gpu_memory_fraction', None, '', lower_bound=0, upper_bound=1)
flags.DEFINE_boolean('profile', False, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('trace', False, '')


def main(args):
    if FLAGS.resume:
        output_dir = path.join(FLAGS.run_dir, FLAGS.resume)
        gin.parse_config_file(path.join(output_dir, 'operative_config-0.gin'))
    else:
        run_name = FLAGS.run_name or time.strftime('%Y%m%d-%H%M%S', time.localtime())
        output_dir = path.join(FLAGS.run_dir, run_name)
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
                                               save_summaries_secs=60,
                                               save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                                               save_checkpoint_steps=FLAGS.save_checkpoint_steps) as sess:

            summary_writer = tf.summary.FileWriterCache.get(output_dir)
            episode_rewards = deque(maxlen=100)
            last_reward_log_time = 0

            env.start()

            while not sess.should_stop():
                obs, _, episode_end = env.reset()

                episode_sum_reward = 0
                while not sess.should_stop() and not episode_end:
                    def step_fn(step_context):
                        action = actor.get_action(step_context, obs[0])

                        next_obs, rewards, episode_end = env.step([action])

                        actor.receive_reward(step_context, obs[0], action, rewards[0],
                                             next_obs[0], episode_end)

                        nonlocal episode_sum_reward, last_reward_log_time
                        episode_sum_reward += rewards[0]

                        if time.time() - last_reward_log_time >= 60 and len(episode_rewards) > 0:
                            global_step = step_context.session.run(tf.train.get_global_step())
                            rewards_np = np.asarray(episode_rewards)
                            summary_writer.add_summary(
                                tf.Summary(value=[
                                    tf.Summary.Value(tag='episode_reward/mean', simple_value=rewards_np.mean()),
                                    tf.Summary.Value(tag='episode_reward/min', simple_value=rewards_np.min()),
                                    tf.Summary.Value(tag='episode_reward/max', simple_value=rewards_np.max()),
                                    tf.Summary.Value(tag='episode_reward/stdev', simple_value=rewards_np.std())
                                ]),
                                global_step=global_step
                            )
                            print('UpdateStep', global_step, 'Rmin', rewards_np.min(), 'Rmax', rewards_np.max(), 'Rmean', rewards_np.mean(), 'Rstd', rewards_np.std())
                            last_reward_log_time = time.time()

                        return next_obs, episode_end

                    obs, episode_end = sess.run_step_fn(step_fn)

                episode_rewards.append(episode_sum_reward)


if __name__ == '__main__':
    app.run(main)
