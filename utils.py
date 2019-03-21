import time
import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.train import SessionRunHook, SessionRunArgs, SecondOrStepTimer, get_global_step


def print_parameter_summary():
    trainable_vars = []
    total_parameters = 0
    for variable in tf.trainable_variables():
        parameters = np.prod(variable.get_shape().as_list())
        total_parameters += parameters
        trainable_vars.append((variable.name, parameters))

    trainable_vars.sort(key=lambda x: x[1], reverse=True)

    print('List of trainable variables:')
    print('-' * 80)
    for name, params in trainable_vars:
        print('{:70} {}'.format(name, params))

    print('-' * 80)
    print('Total trainable parameters:', total_parameters)


class LogProgressHook(SessionRunHook):

    def __init__(self, step_limit, log_interval_secs=10):
        self.step_limit = step_limit
        self.log_interval_secs = log_interval_secs
        self.global_step = None
        self.start_time = 0
        self.last_log_global_step = 0
        self.last_log_time = 0

    def begin(self):
        self.start_time = time.time()
        self.global_step = get_global_step()
        self.last_log_time = self.start_time

    def after_run(self, run_context, run_values):
        now = time.time()
        if now >= self.last_log_time + self.log_interval_secs:
            current_step = run_context.session.run(self.global_step)
            elapsed_time = now - self.start_time

            steps_per_sec = current_step / elapsed_time
            remaining_secs = (self.step_limit - current_step) / steps_per_sec
            td = datetime.timedelta(seconds=remaining_secs)

            logging.info('{}/{} steps, {:.1f} steps/sec, remaining time: {}'.format(current_step, self.step_limit, steps_per_sec, td))

            self.last_log_global_step = current_step
            self.last_log_time = now
