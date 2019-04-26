import time
import logging
from os import path
from collections import deque
from absl import app, flags
import numpy as np
import gin
import gin.tf
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from .environments import VecEnv, SC2Environment, SC2EnvironmentConfig
from .agents import A2CAgent
from .training import Runner, RewardSummaryHook
from .utils import print_parameter_summary, LogProgressHook

FLAGS = flags.FLAGS
flags.DEFINE_integer('save_checkpoint_secs', None, '')
flags.DEFINE_integer('save_checkpoint_steps', None, '')
flags.DEFINE_integer('step_limit', None, '', lower_bound=0)
flags.DEFINE_string('run_name', None, '')
flags.DEFINE_string('run_dir', 'runs', '')
flags.DEFINE_multi_string('gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', None, 'Newline separated list of Gin parameter bindings.')
flags.DEFINE_string('map', 'MoveToBeacon', '')
flags.DEFINE_boolean('gpu_memory_allow_growth', False, '')
flags.DEFINE_float('gpu_memory_fraction', None, '', lower_bound=0, upper_bound=1)
flags.DEFINE_boolean('profile', False, '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('trace', False, '')

flags.DEFINE_alias('s', 'save_checkpoint_secs')
flags.DEFINE_alias('l', 'step_limit')
flags.DEFINE_alias('n', 'run_name')
flags.DEFINE_alias('d', 'run_dir')
flags.DEFINE_alias('m', 'map')


def main(args):
    run_name = FLAGS.run_name or time.strftime('%Y%m%d-%H%M%S', time.localtime())
    output_dir = path.join(FLAGS.run_dir, run_name)

    gin.bind_parameter('SC2EnvironmentConfig.map_name', FLAGS.map)

    gin_files = []
    if path.exists(output_dir):
        print('Resuming', output_dir)
        gin_files.append(path.join(output_dir, 'operative_config-0.gin'))

    if FLAGS.gin_file:
        gin_files += FLAGS.gin_file

    gin.parse_config_files_and_bindings(gin_files, FLAGS.gin_param, finalize_config=True)

    env = VecEnv(SC2Environment, SC2EnvironmentConfig())
    try:
        agent = A2CAgent(env.spec, callbacks=RewardSummaryHook(summary_output_dir=output_dir, write_summaries_secs=30))
        runner = Runner(env, agent)

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
            hooks.append(tf.train.NanTensorHook(agent.loss))
        with tf.train.MonitoredTrainingSession(config=config, hooks=hooks, checkpoint_dir=output_dir,
                                               save_summaries_secs=30,
                                               save_checkpoint_secs=FLAGS.save_checkpoint_secs,
                                               save_checkpoint_steps=FLAGS.save_checkpoint_steps) as sess:
            while not sess.should_stop():
                def step_fn(step_context):
                    runner.train(step_context, 512)

                sess.run_step_fn(step_fn)
    finally:
        env.close()


if __name__ == '__main__':
    app.run(main)
