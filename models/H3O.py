from itertools import permutations

import tensorflow as tf
import numpy as np

from models.ginv import sigmaPi, prepare_permutation_matices
from utils.other import partitionfunc, groupAvereaging, apply_layers

tf.enable_eager_execution()


class GroupInvariance(tf.keras.Model):
    def __init__(self, perm, num_features=64):
        super(GroupInvariance, self).__init__()
        activation = tf.keras.activations.tanh
        tanh_p1 = lambda x: tf.keras.activations.tanh(x) + 1.

        self.num_features = num_features
        self.n = len(perm[0])
        self.m = len(perm)
        self.p = prepare_permutation_matices(perm, self.n, self.m)

        self.features = [
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(self.n * 64, tanh_p1),
        ]

        self.fc = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x):
        x = apply_layers(x, self.features)
        x = tf.reshape(x, (-1, self.n, self.num_features, self.n))
        x = sigmaPi(x, self.m, self.n, self.p)
        x = apply_layers(x, self.fc)
        return x


class FC(tf.keras.Model):
    def __init__(self):
        super(FC, self).__init__()
        activation = tf.keras.activations.tanh

        self.features = [
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(172, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1, None),
        ]

    def call(self, x):
        x = tf.reshape(x, (-1, 18))
        x = apply_layers(x, self.features)
        return x


class FCGroupAvg(tf.keras.Model):
    def __init__(self):
        super(FCGroupAvg, self).__init__()
        activation = tf.keras.activations.tanh

        self.features = [
            # 19k
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(172, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1, None),
        ]

    def call(self, x):
        x1 = tf.reshape(x, (-1, 18))
        x2 = tf.concat([x[:, 0], x[:, 2], x[:, 1], x[:, 4], x[:, 3], x[:, 5]], axis=-1) # (12)
        x3 = tf.concat([x[:, 2], x[:, 1], x[:, 0], x[:, 5], x[:, 4], x[:, 3]], axis=-1) # (13)
        x4 = tf.concat([x[:, 1], x[:, 0], x[:, 2], x[:, 3], x[:, 5], x[:, 4]], axis=-1) # (23)
        x5 = tf.concat([x[:, 2], x[:, 0], x[:, 1], x[:, 4], x[:, 5], x[:, 3]], axis=-1) # (123)
        x6 = tf.concat([x[:, 1], x[:, 2], x[:, 0], x[:, 5], x[:, 3], x[:, 4]], axis=-1) # (132)
        x = tf.stack([x1, x2, x3, x4, x5, x6], axis=1)
        x = apply_layers(x, self.features)
        x = tf.reduce_mean(x, axis=1)
        return x
