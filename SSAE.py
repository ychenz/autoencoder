# Created by Yuchen on 4/23/17.
import tensorflow as tf
import math


class SSAE(object):
    def __init__(self, shape, batch_size, is_training, p=0.9, beta=0.7, lr=0.001):
        '''
        Build a stacked sparse autoencoder
        :param shape: [input_size, encode_layer_size1, encode_layer_size_2 ...]
        :param batch_size:
        :param is_training:
        :param p:
        :param beta:
        :param lr:
        '''

        self._lr = lr
        self._w = {}
        self._b = {}
        self._b_prime = {}
        self._kl_div = {}
        self._sparse_penalty = tf.Variable(tf.zeros(batch_size), dtype=tf.float32, trainable=False)

        if len(shape) < 2:
            raise ValueError("Must at lease have 1 encoding layer!")

        input_size = shape[0]
        with tf.name_scope('SSAE'):
            self._inputs = tf.placeholder(shape=(batch_size, input_size), dtype=tf.float32, name='input_data')
            num_encode_layer = len(shape) - 1
            # encoding #
            last_output = self._inputs
            for i in range(1, num_encode_layer + 1):
                hidden_size = shape[i]
                std = math.sqrt(2.0 / input_size)

                # a forward pass
                self._w[i] = tf.Variable(tf.truncated_normal((input_size, hidden_size), stddev=std), name='w_' + str(i),
                                         dtype=tf.float32)
                self._b[i] = tf.Variable(tf.zeros(hidden_size), name='b_' + str(i), dtype=tf.float32)
                self._b_prime[i] = tf.Variable(tf.zeros(input_size), name='b_prime_' + str(i), dtype=tf.float32)
                layer_output = tf.matmul(last_output, self._w[i]) + self._b[i]
                activated_output = tf.sigmoid(layer_output, name='output_layer_' + str(i))

                # calculating sparsity
                p_hat = tf.reduce_mean(activated_output, 1, name='mean_activate_' + str(i))
                self._kl_div[i] = self.kl_divergence(p, p_hat)
                input_size = hidden_size
                last_output = activated_output

            self._feature_vector = last_output

            if not is_training:
                return

            # decoding
            last_decoded_output = activated_output
            for i in reversed(range(1, num_encode_layer + 1)):
                self._sparse_penalty = tf.add(self._sparse_penalty, beta * self._kl_div[i])
                decoded_output = tf.matmul(last_decoded_output, tf.transpose(self._w[i])) + self._b_prime[i]
                last_decoded_output = decoded_output
            self._loss = tf.reduce_mean(tf.reduce_sum((self._inputs - last_decoded_output) ** 2) + self._sparse_penalty)
            optimizer = tf.train.AdamOptimizer(self._lr)

            # training
            self._train_op = optimizer.minimize(self._loss)
            tf.summary.scalar("Training Loss", self._loss)
            self._summary_op = tf.summary.merge_all()

    def kl_divergence(self, p, p_hat):
        return p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)

    @property
    def inputs(self):
        return self._inputs

    @property
    def feature_vector(self):
        return self._feature_vector

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def summary_op(self):
        return self._summary_op


if __name__ == '__main__':
    ssae = SSAE([300, 100, 50], 5, is_training=True)
