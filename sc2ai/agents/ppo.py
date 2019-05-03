import gin.tf
import tensorflow as tf
from tensorflow.keras.layers import Input

from .actor_critic import ActorCriticAgent


@gin.configurable
class PPOAgent(ActorCriticAgent):
    def __init__(
            self,
            env_spec,
            callbacks=None
    ):
        self.input_old_log_probs = Input(shape=(), name='input_old_log_probs')

        super().__init__(env_spec, callbacks)

    def optimize(self, run_context, returns):
        feed = self.train_feed(self.history.observations, self.history.actions, returns)

        old_log_probs = run_context.session.run(self.model.policy.log_prob, feed_dict=feed)

        feed[self.input_old_log_probs] = old_log_probs

        run_context.run_with_hooks(self.train_op,
                                   feed_dict=feed)

    def build_loss(self):
        return tf.add_n([self.value_loss(), self.policy_loss(), self.entropy_loss()])

    def policy_loss(self):
        with tf.name_scope('policy_loss'):
            advantage = tf.stop_gradient(self.input_returns - self.model.value)

            ratio = tf.exp(self.model.policy.log_prob - self.input_old_log_probs)

            clip_range = 0.2
            policy_loss = -tf.reduce_mean(tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - clip_range, 1 + clip_range) * advantage
            ))

        tf.summary.scalar('policy_loss', policy_loss, family='losses')

        return policy_loss
