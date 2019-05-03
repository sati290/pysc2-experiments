import tensorflow as tf

from .actor_critic import ActorCriticAgent


class A2CAgent(ActorCriticAgent):
    def __init__(
            self,
            env_spec,
            callbacks=None
    ):
        super().__init__(env_spec, callbacks)

    def optimize(self, run_context, returns):
        run_context.run_with_hooks(self.train_op, feed_dict=self.train_feed(self.history.observations, self.history.actions, returns))

    def build_loss(self):
        return tf.add_n([self.value_loss(), self.policy_loss(), self.entropy_loss()])

    def policy_loss(self):
        with tf.name_scope('policy_loss'):
            advantage = self.input_returns - self.model.value
            policy_loss = -tf.reduce_mean(self.model.policy.log_prob * tf.stop_gradient(advantage))

        tf.summary.scalar('policy_loss', policy_loss, family='losses')

        return policy_loss
