import numpy as np
import gin
import gin.tf
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Flatten, Dense, Conv2D

from .common import preprocess_spatial_observation
from .policy import Policy


@gin.configurable
def spatial_stream(obs, spec, conv_filters=(16, 32), conv_kernel_size=(5, 3)):
    x = preprocess_spatial_observation(obs, spec)
    x = Concatenate(1)(x)
    for f, k in zip(conv_filters, conv_kernel_size):
        x = Conv2D(f, k, padding='same', activation='linear', data_format='channels_first', kernel_initializer=tf.keras.initializers.Orthogonal())(x)

    return x


def logits_output(name, spec, dense, spatial, available_actions):
    with tf.name_scope(name):
        if spec.obs_space:
            logits = Conv2D(1, 1, activation='linear', data_format='channels_first', kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.1))(spatial[spec.obs_space])
            logits = Flatten()(logits)
        else:
            logits = Dense(np.prod(spec.sizes), activation='linear',
                           kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.1))(dense)

        if name == 'function_id':
            logits = tf.where(available_actions > 0, logits, -1000 * tf.ones_like(logits), name='mask_unavailable_functions')

    return logits


@gin.configurable
def value_output(state, activation='linear'):
    x = Dense(1, activation=activation, kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.1))(state)
    return tf.squeeze(x)


@gin.configurable
class FullyConvModel:
    def __init__(self, observations, actions_taken, env_spec):
        with tf.name_scope('fully_conv_model'):
            spatial_streams = {name: spatial_stream(observations[name], spec)
                               for name, spec in env_spec.observation_spec.items() if spec.is_spatial}

            fc = Concatenate()([Flatten()(x) for x in spatial_streams.values()])
            fc = Dense(256, activation='relu', name='fc', kernel_initializer=tf.keras.initializers.Orthogonal())(fc)

            with tf.name_scope('logits'):
                self.logits = {name: logits_output(name, spec, fc, spatial_streams, observations['available_actions'])
                               for name, spec in env_spec.action_spec.items()}

            self.policy = Policy(self.logits, actions_taken, env_spec.action_spec)

            with tf.name_scope('value'):
                self.value = value_output(fc)
