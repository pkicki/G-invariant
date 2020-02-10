from itertools import permutations

import tensorflow as tf
import numpy as np

from models.ginv import sigmaPi, prepare_permutation_matices
from utils.other import partitionfunc, groupAvereaging

tf.enable_eager_execution()


class GroupInvariance(tf.keras.Model):
    def __init__(self, perm, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvariance, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(4 * num_features, None),
        ]

        self.fc = [
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, self.n, -1, self.n))
        x = sigmaPi(x, self.m, self.n, self.p)
        for layer in self.fc:
            x = layer(x)
        return x


class GroupInvarianceConv(tf.keras.Model):
    def __init__(self, perm, num_features):
        super(GroupInvarianceConv, self).__init__()

        activation = tf.keras.activations.tanh
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(4 * num_features, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1),
        ]

        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

    def call(self, x, training=None):
        bs = x.shape[0]
        x = tf.concat([x[:, -1:], x, x[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, self.n, -1, self.n))
        x = sigmaPi(x, self.m, self.n, self.p)
        for layer in self.fc:
            x = layer(x)
        return x


class Conv1d(tf.keras.Model):
    def __init__(self, activation=tf.keras.activations.tanh):
        super(Conv1d, self).__init__()
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(2, 1, padding='same', activation=activation),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, x):
        bs = x.shape[0]
        x = tf.concat([x[:, -1:], x, x[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        for layer in self.fc:
            x = layer(x)
        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        return x


class SimpleNet(tf.keras.Model):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = [
            tf.keras.layers.Dense(64, tf.keras.activations.tanh),
            tf.keras.layers.Dense(18, tf.keras.activations.tanh),
            tf.keras.layers.Dense(1),
        ]

    def process(self, quad):
        x = tf.reshape(quad, (-1, 8))
        for layer in self.features:
            x = layer(x)

        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        return x


class MulNet(tf.keras.Model):
    def __init__(self):
        super(MulNet, self).__init__()
        self.fc = [
            tf.keras.layers.Dense(32, tf.keras.activations.tanh),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x):
        for l in self.fc:
            x = l(x)
        return x


class Maron(tf.keras.Model):
    def __init__(self):
        super(Maron, self).__init__()

        self.features = [
            tf.keras.layers.Dense(40, tf.keras.activations.tanh),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        self.a = list(set([p for x in partitionfunc(4, 8, l=0) for p in permutations(x)]))
        self.f = np.array(self.a)

    def call(self, x, training=None):
        def inv(a, b, c, d, e, f, g, h):
            p = self.f
            x1 = a ** p[:, 0]
            x2 = b ** p[:, 1]
            x3 = c ** p[:, 2]
            x4 = d ** p[:, 3]
            x5 = e ** p[:, 4]
            x6 = f ** p[:, 5]
            x7 = g ** p[:, 6]
            x8 = h ** p[:, 7]
            mul = x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8
            mulnn = self.mulnn(tf.stack([x1, x2, x3, x4, x5, x6, x7, x8], axis=-1))[:, :, 0]
            mul_loss = tf.keras.losses.mean_absolute_error(mul, mulnn)
            return mulnn, mul_loss

        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, 8))
        a, b, c, d, e, f, g, h = tf.unstack(x[:, :, tf.newaxis], axis=1)

        def term():
            p1, l1 = inv(a, b, c, d, e, f, g, h)
            p2, l2 = inv(d, a, b, c, h, e, f, g)
            p3, l3 = inv(c, d, a, b, g, h, e, f)
            p4, l4 = inv(b, c, d, a, f, g, h, e)
            q1 = p1 + p2 + p3 + p4
            L = l1 + l2 + l3 + l4
            return q1, L

        x, L = term()

        for layer in self.features:
            x = layer(x)

        return x, L
