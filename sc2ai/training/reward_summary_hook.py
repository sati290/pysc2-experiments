import time
from collections import deque
import numpy as np
import tensorflow as tf


class RewardSummaryHook:
    def __init__(self, summary_writer=None, write_summaries_secs=60):
        self.summary_writer = summary_writer
        self.write_summaries_secs = write_summaries_secs

        self.last_summary_time = 0
        self.sum_episode_rewards = None
        self.episode_rewards = deque(maxlen=100)

    def on_step(self, rewards, episode_ends):
        if self.sum_episode_rewards is None:
            self.sum_episode_rewards = np.zeros(len(rewards), dtype=np.float32)
        self.sum_episode_rewards += rewards

        for i, end in enumerate(episode_ends):
            if end:
                self.episode_rewards.append(self.sum_episode_rewards[i])
                self.sum_episode_rewards[i] = 0

    def on_update(self, global_step):
        if self.summary_writer and time.time() - self.last_summary_time > self.write_summaries_secs and len(self.episode_rewards) > 0:
            rewards_np = np.asarray(self.episode_rewards)
            self.summary_writer.add_summary(
                tf.Summary(value=[
                    tf.Summary.Value(tag='episode_reward/mean', simple_value=rewards_np.mean()),
                    tf.Summary.Value(tag='episode_reward/min', simple_value=rewards_np.min()),
                    tf.Summary.Value(tag='episode_reward/max', simple_value=rewards_np.max()),
                    tf.Summary.Value(tag='episode_reward/stdev', simple_value=rewards_np.std())
                ]),
                global_step=global_step
            )

            self.last_summary_time = time.time()
