import numpy as np
import gin
import gin.tf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Lambda, RNN, LSTMCell, Softmax
from pysc2.lib.features import FeatureType


@gin.configurable
def input_block(features, space_desc, conv_features=(8, 4), conv_activation='linear'):
    conv = Conv2D(conv_features[space_desc.index], 1, data_format='channels_first', activation=conv_activation,
                  name=space_desc.name + '_conv')

    features = Concatenate(axis=1, name=space_desc.name + '_concat_inputs')(features)
    features = conv(features)

    tf.summary.histogram('{}_conv_out'.format(space_desc.name), features)
    tf.summary.histogram('{}_conv_weights'.format(space_desc.name), conv.weights[0])

    return features


@gin.configurable
def predictions_output(state, space_desc, spatial_features=(8, 4), spatial_activation='elu',
                       output_activation='linear'):
    spatial_shape = (spatial_features[space_desc.index],) + space_desc.shape[1:]
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
    def probability_output(num_categories, name, mask=None):
        x = Dense(num_categories, activation='linear', name=name + '_logits')(state)
        if mask is not None:
            x = tf.where(mask > 0, x, -1000 * tf.ones_like(x), name=name + '_mask')
        return tfp.distributions.Categorical(logits=x, name=name + '_dist')

    fn_dist = probability_output(len(action_spec.functions), mask=available_actions, name='fn')

    arg_dists = {
        arg.name: probability_output(np.prod(arg.sizes), name='arg_{}'.format(arg.name))
        for arg in action_spec.types
    }

    return fn_dist, arg_dists


@gin.configurable
def sample_policy(policy):
    fn_dist, arg_dists = policy

    fn_sample = fn_dist.sample(name='fn_sample')
    arg_samples = {k: v.sample(name='arg_{}_sample'.format(k)) for k, v in arg_dists.items()}

    return fn_sample, arg_samples


@gin.configurable
class Model:
    def __init__(self, spatial_features, available_actions, env_spec, dense_layer_size=(512,),
                 activation='elu', output_predictions_fn=None):

        with tf.name_scope('model'):
            with tf.name_scope('input'):
                spatial_features = [input_block(spatial_features[s.index], s) for s in env_spec.spaces]

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
                self.policy = policy_output(dense, available_actions, env_spec.action_spec)

            with tf.name_scope('actions'):
                self.actions = sample_policy(self.policy)

            if output_predictions_fn:
                with tf.name_scope('prediction'):
                    self.prediction = [output_predictions_fn(dense, s) for s in env_spec.spaces]
