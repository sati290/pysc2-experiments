import gin.tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input

from ..models import FullyConvModel
from .history import History


@gin.configurable(blacklist=('callbacks',))
class ActorCriticAgent:
    def __init__(
            self,
            env_spec,
            callbacks=None,
            model_class=FullyConvModel,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0005,
            discount=0.99,
            trajectory_length=16,
            batch_size=32,
            max_grads_norm=100,
            entropy_factor=0.0005,
            value_factor=0.5
    ):
        self.callbacks = callbacks
        self.discount = discount
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor

        self.input_observations = {name: Input(shape=spec.shape, name='input_{}'.format(name)) for name, spec in env_spec.observation_spec.items()}
        self.input_actions = {name: Input(shape=(), name='input_arg_{}_value'.format(name), dtype='int32') for name in env_spec.action_spec}
        self.input_returns = Input(shape=(), name='input_returns')

        self.model = model_class(self.input_observations, self.input_actions, env_spec)

        self.loss = self.build_loss()

        self.optimizer = optimizer(learning_rate=learning_rate)
        grads, vars = zip(*self.optimizer.compute_gradients(self.loss))
        grads_norm = tf.global_norm(grads)
        if max_grads_norm > 0:
            grads, _ = tf.clip_by_global_norm(grads, max_grads_norm, grads_norm)
        self.train_op = self.optimizer.apply_gradients(zip(grads, vars), global_step=tf.train.get_or_create_global_step())

        self.history = History(trajectory_length, batch_size, env_spec)

        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', self.loss, family='losses')
        tf.summary.scalar('grads_norm', grads_norm)

    def get_action(self, run_context, obs):
        return run_context.session.run(self.model.policy.sample, feed_dict=self.obs_feed(obs))

    def on_step(self, run_context, obs, action, reward, next_obs, episode_end):
        self.history.append(obs, action, reward, next_obs, episode_end)

        if self.callbacks:
            self.callbacks.on_step(reward, episode_end)

        if self.history.batch_ready():
            self.train(run_context)
            self.history.reset()

    def train(self, run_context):
        last_values = run_context.session.run(self.model.value, feed_dict=self.obs_feed(self.history.last_observations))

        returns = np.zeros_like(self.history.rewards)
        for i in reversed(range(self.history.trajectory_length)):
            next_values = returns[i + 1] if i + 1 < self.history.trajectory_length else last_values
            discounts = self.discount * (1 - self.history.episode_ends[i])
            returns[i] = self.history.rewards[i] + discounts * next_values

        self.optimize(run_context, returns)

        if self.callbacks:
            global_step = run_context.session.run(tf.train.get_global_step())
            self.callbacks.on_update(global_step)

    def optimize(self, run_context, returns):
        ...

    def obs_feed(self, obs):
        return {input_obs: np.array(obs[name], ndmin=input_obs.shape.rank, copy=False) for name, input_obs in self.input_observations.items()}

    def train_feed(self, obs, actions, returns):
        obs = {k: v.reshape((-1,) + v.shape[2:]) for k, v in obs.items()}
        actions = {k: v.reshape((-1,) + v.shape[2:]) for k, v in actions.items()}
        returns = returns.reshape((-1,))

        fd = self.obs_feed(obs)

        for k, v in self.input_actions.items():
            fd[v] = actions[k]

        fd[self.input_returns] = returns

        return fd

    def build_loss(self):
        ...

    def value_loss(self):
        with tf.name_scope('value_loss'):
            loss = tf.losses.mean_squared_error(self.model.value, self.input_returns) * self.value_factor

        tf.summary.scalar('value_loss', loss, family='losses')

        return loss

    def entropy_loss(self):
        with tf.name_scope('entropy_loss'):
            entropy = tf.reduce_mean(self.model.policy.entropy)
            entropy_loss = -entropy * self.entropy_factor

        tf.summary.scalar('policy_entropy', entropy, family='entropy')
        tf.summary.scalar('entropy_loss', entropy_loss, family='losses')

        return entropy_loss
