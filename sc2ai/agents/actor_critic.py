import gin.tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from pysc2.lib.features import FeatureType

from ..models import FullyConvModel
from .history import History


@gin.configurable
class A2CAgent:
    def __init__(
            self,
            env_spec,
            model_class=FullyConvModel,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.0001,
            discount=0.99,
            trajectory_length=16,
            batch_size=32,
            max_grads_norm=100,
            policy_factor=1,
            entropy_factor=0.0001,
            value_factor=0.5
    ):
        self.discount = discount
        self.env_spec = env_spec
        self.policy_factor = policy_factor
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor

        self.input_observations = {name: Input(shape=spec.shape, name='input_{}'.format(name)) for name, spec in env_spec.observation_spec.items()}
        self.input_actions = {name: Input(shape=(), name='input_arg_{}_value'.format(name), dtype='int32') for name in env_spec.action_spec}
        self.input_returns = Input(shape=(), name='input_returns')

        self.function_args_mask = tf.constant(env_spec.action_spec['function_id'].args_mask, dtype=tf.float32,
                                              name='function_args_mask')

        self.model = model_class(self.input_observations, env_spec)

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
        return run_context.session.run(self.model.actions, feed_dict=self.obs_feed(obs))

    def receive_reward(self, run_context, obs, action, reward, next_obs, episode_end):
        if self.history.append(obs, action, reward, next_obs, episode_end):
            self.train(run_context)

    def train(self, run_context):
        last_values = run_context.session.run(self.model.value, feed_dict=self.obs_feed(self.history.last_observations))

        returns = np.zeros_like(self.history.rewards)
        for i in reversed(range(self.history.trajectory_length)):
            next_values = returns[i + 1] if i + 1 < self.history.trajectory_length else last_values
            discounts = self.discount * (1 - self.history.episode_ends[i])
            returns[i] = self.history.rewards[i] + discounts * next_values

        run_context.run_with_hooks(self.train_op, feed_dict=self.train_feed(self.history.observations, self.history.actions, returns))

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
        return tf.add_n([self.value_loss(), self.policy_loss(), self.entropy_loss()])

    def value_loss(self):
        with tf.name_scope('value_loss'):
            loss = tf.losses.mean_squared_error(self.model.value, self.input_returns) * self.value_factor

        tf.summary.scalar('value_loss', loss, family='losses')

        return loss

    def policy_loss(self):
        with tf.name_scope('policy_loss'):
            log_probs = [dist.log_prob(self.input_actions[name]) for name, dist in self.model.policy.items()]
            log_probs = tf.stack(log_probs, axis=-1)
            log_probs = log_probs * tf.gather(self.function_args_mask, self.input_actions['function_id'])

            advantage = self.input_returns - self.model.value

            policy_loss = -tf.reduce_mean(tf.reduce_sum(log_probs, axis=-1) * tf.stop_gradient(advantage)) * self.policy_factor

        tf.summary.scalar('policy_loss', policy_loss, family='losses')

        return policy_loss

    def entropy_loss(self):
        with tf.name_scope('entropy_loss'):
            entropies = [dist.entropy() for name, dist in self.model.policy.items()]
            entropy = tf.reduce_mean(tf.add_n(entropies))
            entropy_loss = -entropy * self.entropy_factor

        entropy_masked = tf.stack(entropies, axis=-1) * tf.gather(self.function_args_mask, self.input_actions['function_id'])
        entropy_masked = tf.reduce_mean(tf.reduce_sum(entropy_masked, axis=-1))
        tf.summary.scalar('policy_entropy', entropy, family='entropy')
        tf.summary.scalar('policy_entropy_masked', entropy_masked, family='entropy')
        tf.summary.scalar('entropy_loss', entropy_loss, family='losses')

        return entropy_loss


class A2CPredictionAgent(A2CAgent):
    def __init__(self, env_spec):
        super().__init__(env_spec)

        loss_pred = self.prediction_loss(spatial_features, feat_palettes)

    def preprocess_observations(self):
        def one_hot_encode(x, scale):
            x = tf.squeeze(x, axis=1)
            x = tf.cast(x, tf.int32)
            return tf.one_hot(x, scale, axis=1)

        def preprocess_observation(input_obs, spec):
            if spec.is_spatial:
                features = Lambda(lambda x: tf.split(x, x.get_shape()[1], axis=1))(input_obs)

                for f in spec.features:
                    if f.type == FeatureType.CATEGORICAL:
                        features[f.index] = Lambda(lambda x: one_hot_encode(x, f.scale))(features[f.index])
                    else:
                        features[f.index] = Lambda(lambda x: x / f.scale)(features[f.index])

                return features
            else:
                return input_obs

        with tf.name_scope('preprocess_observations'):
            return {name: preprocess_observation(self.input_observations[name], spec)
                    for name, spec in self.env_spec.observation_spec.items()}

    def prediction_loss(self, truths, palette):
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
            for s in self.env_spec.spaces:
                with tf.name_scope(s.name):
                    loss = spatial_loss(truths[s.index], self.out_pred[s.index], s)
                    spatial_losses.append(loss)
                    tf.summary.scalar('loss', loss)

            loss = tf.reduce_mean(tf.stack(spatial_losses))
            tf.summary.scalar('loss', loss)

        return loss
