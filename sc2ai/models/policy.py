import tensorflow as tf
import tensorflow_probability as tfp


class Policy:
    def __init__(self, logits, actions_taken, action_spec):
        with tf.name_scope('policy'):
            self.function_args_mask = tf.constant(action_spec['function_id'].args_mask, dtype=tf.float32,
                                                  name='function_args_mask')

            self.dists = {k: tfp.distributions.Categorical(logits=v, name=k + 'dist') for k, v in logits.items()}

            self.sample = {name: dist.sample(name=name + '_sample') for name, dist in self.dists.items()}
            self.entropy = tf.add_n([dist.entropy(name=name + '_entropy') for name, dist in self.dists.items()])

            log_probs = [dist.log_prob(actions_taken[name], name=name + 'log_prob') for name, dist in self.dists.items()]
            log_probs = tf.stack(log_probs, axis=-1)
            log_probs = log_probs * tf.gather(self.function_args_mask, actions_taken['function_id'])
            self.log_prob = tf.reduce_sum(log_probs, axis=-1)
