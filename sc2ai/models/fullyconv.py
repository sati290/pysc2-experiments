import numpy as np
import gin
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Concatenate, Flatten, Dense, Conv2D

from .common import preprocess_spatial_observation


def spatial_stream(obs, spec):
    x = preprocess_spatial_observation(obs, spec)
    x = Concatenate(1)(x)
    x = Conv2D(16, 1, activation='linear', data_format='channels_first')(x)

    return x


@gin.configurable
def value_output(state, activation='linear'):
    x = Dense(1, activation=activation)(state)
    return tf.squeeze(x)


@gin.configurable
class FullyConvModel:
    def __init__(self, observations, env_spec):
        with tf.name_scope('fully_conv_model'):
            spatial_streams = {name: spatial_stream(observations[name], spec)
                               for name, spec in env_spec.observation_spec.items() if spec.is_spatial}

            fc = Concatenate()([Flatten()(x) for x in spatial_streams.values()])
            fc = Dense(256, activation='relu')(fc)

            with tf.name_scope('policy'):
                self.policy = {}
                for name, spec in env_spec.action_spec.items():
                    with tf.name_scope(name):
                        if spec.obs_space:
                            logits = Conv2D(1, 1, activation='linear', data_format='channels_first')(spatial_streams[spec.obs_space])
                            logits = Flatten()(logits)
                        else:
                            logits = Dense(np.prod(spec.sizes), activation='linear')(fc)

                        if name == 'function_id':
                            logits = tf.where(observations['available_actions'] > 0, logits, -1000 * tf.ones_like(logits), name='mask_unavailable_functions')

                        self.policy[name] = tfp.distributions.Categorical(logits=logits)

            with tf.name_scope('actions'):
                self.actions = {name: dist.sample(name=name + '_sample') for name, dist in self.policy.items()}

            with tf.name_scope('value'):
                self.value = value_output(fc)
