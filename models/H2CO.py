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
            #tf.keras.layers.Dense(128, activation),
            #tf.keras.layers.Dense(32, activation),
            #tf.keras.layers.Dense(256, activation, trainable=False),
            #tf.keras.layers.Dense(64, tanh_p1),
            tf.keras.layers.Dense(32, activation),
            #tf.keras.layers.Dense(self.n * self.num_features, None),
            #tf.keras.layers.Dense(self.n * self.num_features, activation),
            tf.keras.layers.Dense(self.n * self.num_features, tanh_p1),
            #tf.keras.layers.Dense(self.n * 64, tanh_p1),
            #tf.keras.layers.Dense(self.n * self.num_features, tanh_p1, trainable=False),
        ]

        self.fc = [
            #tf.keras.layers.Dense(256, tf.keras.activations.relu),
            #tf.keras.layers.Dense(128, tf.keras.activations.relu),
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

            #tf.keras.layers.Dense(128, activation),
            #tf.keras.layers.Dense(64, activation),
            #tf.keras.layers.Dense(1, None),

            #tf.keras.layers.Dense(256, activation),
            #tf.keras.layers.Dense(1024, activation),
            #tf.keras.layers.Dense(1, None),
        ]

    def call(self, x):
        #x = tf.reshape(x, (-1, 6))
        x = tf.reshape(x, (-1, 18))
        x = apply_layers(x, self.features)
        return x


class FCGroupAvg(tf.keras.Model):
    def __init__(self):
        super(FCGroupAvg, self).__init__()
        activation = tf.keras.activations.tanh

        self.features = [
            # wide
            # tf.keras.layers.Dense(512, activation),
            # tf.keras.layers.Dense(256, activation),
            # tf.keras.layers.Dense(128, activation),

            #tf.keras.layers.Dense(256, activation),
            #tf.keras.layers.Dense(1024, activation),
            #tf.keras.layers.Dense(1, None),

            # big
            # tf.keras.layers.Dense(512, activation),
            # tf.keras.layers.Dense(256, activation),
            # tf.keras.layers.Dense(128, activation),
            # tf.keras.layers.Dense(64, activation),
            # tf.keras.layers.Dense(16, activation),
            # tf.keras.layers.Dense(1, None),

            # _
            #tf.keras.layers.Dense(128, activation),
            #tf.keras.layers.Dense(64, activation),
            #tf.keras.layers.Dense(1, None),

            # 19k
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(172, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1, None),
        ]

    def call(self, x):
        x1 = x
        #x2 = tf.stack([x[:, 0], x[:, 2], x[:, 1], x[:, 4], x[:, 3], x[:, 5]], axis=1)
        x2 = tf.stack([x[:, 1], x[:, 0], x[:, 2], x[:, 3]], axis=1)
        #x1 = tf.reshape(x1, (-1, 18))
        #x1 = tf.reshape(x1, (-1, 6))
        x1 = tf.reshape(x1, (-1, 12))
        x1 = apply_layers(x1, self.features)
        #x2 = tf.reshape(x2, (-1, 18))
        #x2 = tf.reshape(x2, (-1, 6))
        x2 = tf.reshape(x2, (-1, 12))
        x2 = apply_layers(x2, self.features)
        x = (x1 + x2) / 2.
        return x


class MulNet(tf.keras.Model):
    def __init__(self):
        super(MulNet, self).__init__()
        activation = tf.keras.activations.tanh

        self.fc = [
            tf.keras.layers.Dense(256, activation),
            tf.keras.layers.Dense(128, activation),
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
            tf.keras.layers.Dense(512, activation),
            tf.keras.layers.Dense(256, activation),
            tf.keras.layers.Dense(128, activation),
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        # Z5 in S5
        self.a = list(set([p for x in partitionfunc(2, 6, l=0) for p in permutations(x)]))
        self.f = np.array(self.a)

    def call(self, x):
        def inv(a, b, c, d, e, f):
            p = self.f
            x1 = a ** p[:, 0]
            x2 = b ** p[:, 1]
            x3 = c ** p[:, 2]
            x4 = d ** p[:, 3]
            x5 = e ** p[:, 4]
            x6 = f ** p[:, 5]
            mul = x1 * x2 * x3 * x4 * x5 * x6
            mulnn = self.mulnn(tf.stack([x1, x2, x3, x4, x5, x6], axis=-1))[:, :, 0]
            mul_loss = tf.keras.losses.mean_absolute_error(mul, mulnn)
            return mulnn, mul_loss

        a, b, c, d, e, f = tf.unstack(x, axis=1)

        def term():
            # Z5 in S5
            p1, l1 = inv(a, b, c, d, e, f)
            p2, l2 = inv(a, c, b, e, d, f)
            q1 = p1 + p2
            L = (l1 + l2) / 5.
            return q1, L

        x, L = term()

        x = apply_layers(x, self.features)

        return x, L


class Pooling(tf.keras.Model):
    def __init__(self):
        super(Pooling, self).__init__()
        activation = tf.keras.activations.tanh

        self.hh = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
        ]

        self.co = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
        ]

        self.hco = [
            tf.keras.layers.Dense(128, activation),
            tf.keras.layers.Dense(64, activation),
        ]

        self.features = [
            tf.keras.layers.Dense(40, activation),
            tf.keras.layers.Dense(1, None),
        ]

    def call(self, x):
        x0 = x[:, 0]
        x1 = tf.concat([x[:, 1], x[:, 3]], axis=-1)
        x2 = tf.concat([x[:, 2], x[:, 4]], axis=-1)
        x3 = x[:, -1]
        hh = apply_layers(x0, self.hh)
        h1co = apply_layers(x1, self.hco)
        h2co = apply_layers(x2, self.hco)
        co = apply_layers(x3, self.co)
        h12co = h1co + h2co
        x = tf.concat([hh, h12co, co], axis=-1)
        x = apply_layers(x, self.features)
        return x
