import gin.tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from pysc2.lib.features import FeatureType
from models import BasicModel


@gin.configurable
class A2CAgent:
    def __init__(
            self,
            env_spec,
            optimizer=tf.train.AdamOptimizer,
            learning_rate=0.00005,
            discount=0.99,
            batch_size=32,
            policy_factor=1,
            entropy_factor=0.0001,
            value_factor=0.5
    ):
        self.discount = discount
        self.env_spec = env_spec
        self.batch_size = batch_size
        self.policy_factor = policy_factor
        self.entropy_factor = entropy_factor
        self.value_factor = value_factor

        self.input_observations = {name: Input(shape=spec.shape, name='input_{}'.format(name)) for name, spec in env_spec.observation_spec.items()}
        self.input_action_id = Input(shape=(), name='input_action_id')
        self.input_action_args = {arg.name: Input(shape=(), name='input_arg_{}_value'.format(arg.name)) for arg in env_spec.action_spec.types}
        self.input_returns = Input(shape=(), name='input_returns')

        observations = self.preprocess_observations()

        self.model = self.build_model(observations)

        self.loss = self.build_loss()

        self.optimizer = optimizer(learning_rate=learning_rate)
        grads, vars = zip(*self.optimizer.compute_gradients(self.loss))
        grads_norm = tf.global_norm(grads)
        self.train_op = self.optimizer.apply_gradients(zip(grads, vars), global_step=tf.train.get_or_create_global_step())

        self.history = []

        tf.summary.scalar('value', tf.reduce_mean(self.model.value))
        tf.summary.scalar('returns', tf.reduce_mean(self.input_returns))
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', self.loss, family='losses')
        tf.summary.scalar('grads_norm', grads_norm)

    def get_action(self, run_context, obs):
        return run_context.session.run(self.model.actions, feed_dict=self.obs_feed(obs))

    def receive_reward(self, run_context, obs, action, reward, next_obs, episode_end):
        self.history.append({
            'obs': obs,
            'action': action,
            'reward': reward
        })

        if len(self.history) >= self.batch_size or episode_end:
            if episode_end:
                next_value = 0
            else:
                next_value = run_context.session.run(self.model.value, feed_dict=self.obs_feed(next_obs))

            self.train(run_context, next_value)

    def train(self, run_context, next_value):
        returns = np.zeros(len(self.history))
        for i in reversed(range(len(self.history))):
            v_next = returns[i+1] if i < len(self.history) - 1 else next_value
            returns[i] = self.history[i]['reward'] + self.discount * v_next

        _, values = run_context.run_with_hooks((self.train_op, self.model.value), feed_dict=self.train_feed(
            [h['obs'] for h in self.history],
            [h['action'] for h in self.history],
            returns
        ))

        # values = np.array(values, ndmin=1)
        # print('training step', run_context.session.run(tf.train.get_global_step()), 'next value:', next_value)
        # for i, h in enumerate(self.history):
        #     print(i, 'reward:', h['reward'], 'returns:', returns[i], 'value:', values[i], 'advantage:', returns[i] - values[i])

        self.history.clear()

    def obs_feed(self, obs):
        return {input_obs: np.expand_dims(obs[name], 0) for name, input_obs in self.input_observations.items()}

    def train_feed(self, obs, actions, returns):
        fd = {input_obs: np.stack([np.asarray(o[name]) for o in obs])
              for name, input_obs in self.input_observations.items()}

        fd[self.input_action_id] = np.concatenate([a[0] for a in actions])

        used_args = [{arg.name for arg in self.env_spec.action_spec.functions[a[0].item()].args} for a in actions]
        for k, v in self.input_action_args.items():
            fd[v] = np.asarray([
                a[1][k].item() if k in used_args[i] else -1
                for i, a in enumerate(actions)
            ])

        fd[self.input_returns] = returns

        return fd

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

    def build_model(self, observations):
        return BasicModel(observations, self.env_spec)

    def build_loss(self):
        return self.value_loss() + self.policy_loss() + self.entropy_loss()

    def value_loss(self):
        with tf.name_scope('value_loss'):
            loss = tf.losses.mean_squared_error(self.model.value, self.input_returns) * self.value_factor

        tf.summary.scalar('value_loss', loss, family='losses')

        return loss

    def policy_loss(self):
        with tf.name_scope('policy_loss'):
            def masked_arg_log_prob(arg_dist, arg_id, name):
                with tf.name_scope(name + 'log_prob'):
                    clipped_arg_id = tf.maximum(arg_id, 0)
                    log_prob = arg_dist.log_prob(clipped_arg_id)
                    log_prob *= tf.cast(arg_id >= 0, tf.float32)
                return log_prob

            fn_dist, arg_dists = self.model.policy

            log_probs = [fn_dist.log_prob(self.input_action_id)] + [masked_arg_log_prob(v, self.input_action_args[k], k)
                                                                    for k, v in arg_dists.items()]

            advantage = self.input_returns - self.model.value

            policy_loss = -tf.reduce_mean(tf.add_n(log_probs) * tf.stop_gradient(advantage)) * self.policy_factor

        tf.summary.scalar('advantage', tf.reduce_mean(advantage))
        tf.summary.scalar('policy_loss', policy_loss, family='losses')

        return policy_loss

    def entropy_loss(self):
        with tf.name_scope('entropy_loss'):
            fn_dist, arg_dists = self.model.policy

            entropies = [fn_dist.entropy()] + [v.entropy() * tf.cast(self.input_action_args[k] >= 0, tf.float32)
                                               for k, v in arg_dists.items()]

            entropy = tf.reduce_mean(tf.add_n(entropies))
            entropy_loss = -entropy * self.entropy_factor

        tf.summary.scalar('policy_entropy', entropy)
        tf.summary.scalar('entropy_loss', entropy_loss, family='losses')

        return entropy_loss


class A2CPredictionAgent(A2CAgent):
    def __init__(self, env_spec):
        super().__init__(env_spec)

        loss_pred = self.prediction_loss(spatial_features, feat_palettes)

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
