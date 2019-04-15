import numpy as np
import gin
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Lambda, RNN, LSTMCell, Softmax, Embedding, Permute
from pysc2.lib.features import FeatureType

from .common import preprocess_spatial_observation


@gin.configurable
def input_block(obs, name, obs_spec, conv_features=(8, 4), conv_activation='linear'):
    features = preprocess_spatial_observation(obs, obs_spec)

    conv = Conv2D(conv_features[obs_spec.id], 1, data_format='channels_first', activation=conv_activation,
                  name=name + '_conv')

    features = Concatenate(axis=1, name=name + '_concat_inputs')(features)
    features = conv(features)

    tf.summary.histogram('{}_conv_out'.format(name), features)
    tf.summary.histogram('{}_conv_weights'.format(name), conv.weights[0])

    return features


@gin.configurable
def predictions_output(state, space_desc, spatial_features=(8, 4), spatial_activation='elu',
                       output_activation='linear'):
    spatial_shape = (spatial_features[space_desc.id],) + space_desc.shape[1:]
    output_spatial = Dense(np.prod(spatial_shape), activation=spatial_activation)(state)
    output_spatial = Reshape(spatial_shape)(output_spatial)

    output_conv_features = np.sum([f.scale if f.type == FeatureType.CATEGORICAL else 1 for f in space_desc.features])
    output_conv = Conv2D(output_conv_features, 1, data_format='channels_first', activation=output_activation)(
        output_spatial)

    output = [None] * space_desc.shape[0]
    next_feature_index = 0
    for f in space_desc.features:
        name = '{}_{}_output'.format(space_desc.name, f.name)
        output_features = f.scale if f.type == FeatureType.CATEGORICAL else 1
        output[f.index] = Lambda(lambda x: x[:, next_feature_index:next_feature_index + output_features], name=name)(
            output_conv)
        next_feature_index = next_feature_index + output_features

    return output


@gin.configurable
def value_output(state, activation='linear'):
    out = Dense(1, activation=activation)(state)
    return tf.squeeze(out)


@gin.configurable
def policy_output(state, available_actions, action_spec):
    def logits_output(num_categories, name):
        return Dense(num_categories, activation='linear', name=name + '_logits')(state)

    logits = [logits_output(np.prod(spec.sizes), name) for name, spec in action_spec.items()]
    logits[0] = tf.where(available_actions > 0, logits[0], -1000 * tf.ones_like(logits[0]), name='mask_unavailable_functions')

    dists = {
        name: tfp.distributions.Categorical(logits=logits[spec.id], name=name + '_dist')
        for name, spec in action_spec.items()
    }

    return dists


@gin.configurable
def sample_policy(policy):
    return {k: v.sample(name='{}_sample'.format(k)) for k, v in policy.items()}


@gin.configurable
class BasicModel:
    def __init__(self, observations, env_spec, dense_layer_size=(512,),
                 activation='elu', output_predictions_fn=None):

        with tf.name_scope('model'):
            with tf.name_scope('input'):
                spatial_features = [input_block(observations[name], name, spec)
                                    for name, spec in env_spec.observation_spec.items() if spec.is_spatial]

            with tf.name_scope('core'):
                spatial_features = [Flatten()(f) for f in spatial_features]
                spatial_features = Concatenate(name='concatenate_features')(spatial_features)

                dense = spatial_features
                for i, size in enumerate(dense_layer_size):
                    op = Dense(size, activation=activation, name='state_dense_' + str(i))
                    dense = op(dense)

                    tf.summary.histogram('state_dense_' + str(i) + '_kernel_weights', op.weights[0])

                tf.summary.scalar('dense_zero_fraction', tf.nn.zero_fraction(dense))
                tf.summary.histogram('dense_input', spatial_features)
                tf.summary.histogram('dense_output', dense)

            with tf.name_scope('value'):
                self.value = value_output(dense)

            with tf.name_scope('policy'):
                self.policy = policy_output(dense, observations['available_actions'], env_spec.action_spec)

            with tf.name_scope('actions'):
                self.actions = sample_policy(self.policy)

            if output_predictions_fn:
                with tf.name_scope('prediction'):
                    self.prediction = [output_predictions_fn(dense, s)
                                       for s in env_spec.observation_spec.values() if s.is_spatial]
