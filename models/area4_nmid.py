from itertools import permutations
from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from models.ginv import sigmaPi, prepare_permutation_matices
from utils.permutation_groups import Z4

tf.enable_eager_execution()


def groupAvereaging(inputs, operation):
    x = inputs
    x1 = x
    x2 = tf.roll(x, 1, 1)
    x3 = tf.roll(x, 2, 1)
    x4 = tf.roll(x, 3, 1)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)

    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4], -1), -1)
    return x


class GroupInvariance(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvariance, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            # tf.keras.layers.Dense(4 * 64, tf.keras.activations.tanh),
            # tf.keras.layers.Dense(4 * 64, tf.keras.activations.sigmoid),
            tf.keras.layers.Dense(4 * num_features, None),
        ]

        self.fc = [
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, n_points, -1, 4))
        a, b, c, d = tf.unstack(x, axis=1)
        x = a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * d[:, :, 3] \
            + b[:, :, 0] * c[:, :, 1] * d[:, :, 2] * a[:, :, 3] \
            + c[:, :, 0] * d[:, :, 1] * a[:, :, 2] * b[:, :, 3] \
            + d[:, :, 0] * a[:, :, 1] * b[:, :, 2] * c[:, :, 3]

        for layer in self.fc:
            x = layer(x)

        return x


class GroupInvarianceConv(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvarianceConv, self).__init__()

        activation = tf.keras.activations.tanh
        self.num_features = num_features
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(4 * self.num_features, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(1),
        ]

        perm = Z4
        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)

        a, b, c, d = tf.unstack(x, axis=1)

        x = tf.reshape(x, (-1, self.n, self.n, self.num_features))
        x = tf.transpose(x, (0, 1, 3, 2))
        x = sigmaPi(x, self.m, self.n, self.p)

        a = tf.reshape(a, (-1, 4, self.num_features))
        b = tf.reshape(b, (-1, 4, self.num_features))
        c = tf.reshape(c, (-1, 4, self.num_features))
        d = tf.reshape(d, (-1, 4, self.num_features))

        y = a[:, 0] * b[:, 1] * c[:, 2] * d[:, 3] \
            + b[:, 0] * c[:, 1] * d[:, 2] * a[:, 3] \
            + c[:, 0] * d[:, 1] * a[:, 2] * b[:, 3] \
            + d[:, 0] * a[:, 1] * b[:, 2] * c[:, 3]

        for layer in self.fc:
            x = layer(x)

        return x
