from itertools import permutations
from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from models.ginv import prepare_permutation_matices, sigmaPi
from utils.other import apply_layers, partitionfunc

tf.enable_eager_execution()


def groupAvereaging(inputs, operation):
    x = inputs
    a, b, c, d, e = tf.unstack(x, axis=1)

    # Z5 in S5
    x1 = x
    x2 = tf.stack([b, c, d, e, a], axis=1)
    x3 = tf.stack([c, d, e, a, b], axis=1)
    x4 = tf.stack([d, e, a, b, c], axis=1)
    x5 = tf.stack([e, a, b, c, d], axis=1)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)
    x5 = operation(x5)

    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4, x5], -1), -1)
    return x


class GroupInvariance(tf.keras.Model):
    def __init__(self, perm, num_features):
        super(GroupInvariance, self).__init__()
        activation=tf.keras.activations.tanh

        self.num_features = num_features
        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(self.n * self.num_features, tf.keras.activations.sigmoid),
            #tf.keras.layers.Dense(self.n * self.num_features, None),
        ]

        self.fc = [
            #tf.keras.layers.Dense(32, tf.keras.activations.tanh),
            tf.keras.layers.Dense(32, tf.keras.activations.relu, use_bias=False),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs):
        x = inputs[:, :, tf.newaxis]
        x = apply_layers(x, self.features)
        x = tf.reshape(x, (-1, self.n, self.num_features, self.n))
        x = sigmaPi(x, self.m, self.n, self.p)
        x = apply_layers(x, self.fc)
        return x


class SimpleNet(tf.keras.Model):
    def __init__(self):
        super(SimpleNet, self).__init__()
        activation = tf.keras.activations.tanh
        self.features = [
            tf.keras.layers.Dense(89, activation),
            tf.keras.layers.Dense(6 * 32, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, x):
        x = apply_layers(x, self.features)
        return x

    def call(self, inputs):
        x = groupAvereaging(inputs, self.process)
        return x


class GroupInvarianceConv(tf.keras.Model):
    def __init__(self, perm, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvarianceConv, self).__init__()
        activation = tf.keras.activations.tanh

        self.num_features = num_features
        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(self.n * self.num_features, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs):
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)[:, :, tf.newaxis]
        x = apply_layers(x, self.features)
        x = tf.reshape(x, (-1, self.n, self.n, self.num_features))
        x = tf.transpose(x, (0, 1, 3, 2))  # for the compatibility with already trained models
        x = sigmaPi(x, self.m, self.n, self.p)
        x = apply_layers(x, self.fc)
        return x


class Conv1d(tf.keras.Model):
    def __init__(self):
        super(Conv1d, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 118
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(self.last_n, 1, activation=activation),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, x):
        bs = x.shape[0]
        x = tf.concat([x[:, -1:], x, x[:, :1]], axis=1)[:, :, tf.newaxis]
        x = apply_layers(x, self.features)
        x = tf.reshape(x, (bs, -1))
        x = apply_layers(x, self.fc)
        return x

    def call(self, inputs):
        x = groupAvereaging(inputs, self.process)
        return x


class MulNet(tf.keras.Model):
    def __init__(self):
        super(MulNet, self).__init__()
        activation = tf.keras.activations.tanh

        self.fc = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x):
        x = apply_layers(x, self.fc)
        return x


class Maron(tf.keras.Model):
    def __init__(self):
        super(Maron, self).__init__()
        activation = tf.keras.activations.tanh

        self.features = [
            tf.keras.layers.Dense(48, activation),
            tf.keras.layers.Dense(192, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        # Z5 in S5
        self.a = list(set([p for x in partitionfunc(5, 5, l=0) for p in permutations(x)]))
        self.f = np.array(self.a)

    def call(self, x):
        def inv(a, b, c, d, e):
            p = self.f
            x1 = a ** p[:, 0]
            x2 = b ** p[:, 1]
            x3 = c ** p[:, 2]
            x4 = d ** p[:, 3]
            x5 = e ** p[:, 4]
            mul = x1 * x2 * x3 * x4 * x5
            mulnn = self.mulnn(tf.stack([x1, x2, x3, x4, x5], axis=-1))[:, :, 0]
            mul_loss = tf.keras.losses.mean_absolute_error(mul, mulnn)
            return mulnn, mul_loss

        a, b, c, d, e = tf.unstack(x[:, :, tf.newaxis], axis=1)

        def term():
            # Z5 in S5
            p1, l1 = inv(a, b, c, d, e)
            p2, l2 = inv(e, a, b, c, d)
            p3, l3 = inv(d, e, a, b, c)
            p4, l4 = inv(c, d, e, a, b)
            p5, l5 = inv(b, c, d, e, a)
            q1 = p1 + p2 + p3 + p4 + p5
            L = (l1 + l2 + l3 + l4 + l5) / 5.
            return q1, L

        x, L = term()

        x = apply_layers(x, self.features)

        return x, L
