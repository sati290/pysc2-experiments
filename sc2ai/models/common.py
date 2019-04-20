import gin
import gin.tf
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Flatten, Reshape, Conv2D, Lambda, RNN, LSTMCell, Softmax, Embedding, Permute


@gin.configurable
def preprocess_spatial_observation(input_obs, spec, categorical_embedding_dims=16, non_categorical_scaling='log'):
    with tf.name_scope('preprocess_spatial_obs'):
        features = Lambda(lambda x: tf.split(x, x.get_shape()[1], axis=1))(input_obs)

        for f in spec.features:
            if f.is_categorical:
                features[f.index] = Lambda(lambda x: tf.squeeze(x, axis=1))(features[f.index])
                features[f.index] = Embedding(f.scale, categorical_embedding_dims)(features[f.index])
                features[f.index] = Permute((3, 1, 2))(features[f.index])
            else:
                if non_categorical_scaling == 'log':
                    features[f.index] = Lambda(lambda x: tf.log(x + 1e-10))(features[f.index])
                elif non_categorical_scaling == 'normalize':
                    features[f.index] = Lambda(lambda x: x / f.scale)(features[f.index])

    return features
