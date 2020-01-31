from itertools import permutations
from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

tf.enable_eager_execution()


def groupAvereaging(inputs, operation):
    x = inputs
    a, b, c, d, e = tf.unstack(x, axis=1)
    x1 = x
    x2 = tf.stack([b, c, d, e, a], axis=1)
    x3 = tf.stack([c, d, e, a, b], axis=1)
    x4 = tf.stack([d, e, a, b, c], axis=1)
    x5 = tf.stack([e, a, b, c, d], axis=1)
    # x6 = tf.stack([c, a, b, e, d], axis=1)
    # x2 = tf.stack([b, c, a, e, d], axis=1)
    # x3 = tf.stack([c, a, b, d, e], axis=1)
    # x4 = tf.stack([a, b, c, e, d], axis=1)
    # x5 = tf.stack([b, c, a, d, e], axis=1)
    # x6 = tf.stack([c, a, b, e, d], axis=1)

    # x = tf.stack([x1, x2, x3, x4, x5, x6], 1)
    #x = tf.stack([x1, x2, x3, x4, x5], 1)
    #x = operation(x)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)
    x5 = operation(x5)
    # x6 = operation(x6)

    #x = tf.reduce_mean(x, 1)
    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4, x5], -1), -1)
    return x


class GroupInvariance(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvariance, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(5 * 64, tf.keras.activations.sigmoid),
            # tf.keras.layers.Dense(5 * 64),
        ]

        self.fc = [
            # tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(num_features, tf.keras.activations.relu, use_bias=False),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = inputs[:, :, tf.newaxis]
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, n_points, -1, 5))
        a, b, c, d, e = tf.unstack(x, axis=1)
        # x = a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
        #    + b[:, :, 0] * c[:, :, 1] * a[:, :, 2] * e[:, :, 3] * d[:, :, 4] \
        #    + c[:, :, 0] * a[:, :, 1] * b[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
        #    + a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * e[:, :, 3] * d[:, :, 4] \
        #    + b[:, :, 0] * c[:, :, 1] * a[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
        #    + c[:, :, 0] * a[:, :, 1] * b[:, :, 2] * e[:, :, 3] * d[:, :, 4]
        x = a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
            + e[:, :, 0] * a[:, :, 1] * b[:, :, 2] * c[:, :, 3] * d[:, :, 4] \
            + d[:, :, 0] * e[:, :, 1] * a[:, :, 2] * b[:, :, 3] * c[:, :, 4] \
            + c[:, :, 0] * d[:, :, 1] * e[:, :, 2] * a[:, :, 3] * b[:, :, 4] \
            + b[:, :, 0] * c[:, :, 1] * d[:, :, 2] * e[:, :, 3] * a[:, :, 4]

        for layer in self.fc:
            x = layer(x)
        # x = tf.reduce_sum(x, 1, keep_dims=True)

        return x


class SimpleNet(tf.keras.Model):
    def __init__(self, num_features):
        super(SimpleNet, self).__init__()
        activation = tf.keras.activations.tanh
        self.features = [
            tf.keras.layers.Dense(89, activation),
            # tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(6 * num_features, activation),
            tf.keras.layers.Dense(num_features, activation),
            # tf.keras.layers.Dense(1024, activation),
            # tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, x):
        for layer in self.features:
            x = layer(x)

        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        # x = self.process(inputs)
        return x


class GroupInvarianceConv(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvarianceConv, self).__init__()

        activation = tf.keras.activations.tanh
        self.last_n = 116#128
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(5 * self.last_n, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)[:, :, tf.newaxis]
        bs = x.shape[0]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, 5, 5, self.last_n))
        a, b, c, d, e = tf.unstack(x, axis=1)
        #a = tf.reshape(a, (-1, 5, self.last_n))
        #b = tf.reshape(b, (-1, 5, self.last_n))
        #c = tf.reshape(c, (-1, 5, self.last_n))
        #d = tf.reshape(d, (-1, 5, self.last_n))
        #e = tf.reshape(e, (-1, 5, self.last_n))

        x = a[:, 0] * b[:, 1] * c[:, 2] * d[:, 3] * e[:, 4] \
            + b[:, 0] * c[:, 1] * d[:, 2] * e[:, 3] * a[:, 4] \
            + c[:, 0] * d[:, 1] * e[:, 2] * a[:, 3] * b[:, 4] \
            + d[:, 0] * e[:, 1] * a[:, 2] * b[:, 3] * c[:, 4] \
            + e[:, 0] * a[:, 1] * b[:, 2] * c[:, 3] * d[:, 4]

        for layer in self.fc:
            x = layer(x)

        return x


class Conv1d(tf.keras.Model):
    def __init__(self, num_features):
        super(Conv1d, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 118  # 128
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            # tf.keras.layers.Conv1D(64, 3, activation=activation),
            tf.keras.layers.Conv1D(self.last_n, 1, padding='same', activation=activation),
            # tf.keras.layers.Conv1D(self.last_n, 2, activation=activation),
        ]
        # self.fc = tf.keras.layers.Dense(num_features, activation=activation)
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, quad):
        x = quad
        bs = x.shape[0]
        # x = tf.reshape(quad, (-1, 8))
        x = tf.concat([quad[:, -1:], quad, quad[:, :1]], axis=1)[:, :, tf.newaxis]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, -1))
        # x = self.fc(x)
        for layer in self.fc:
            x = layer(x)

        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        # x = self.process(inputs)
        return x




def partitionfunc(n, k, l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l, n + 1):
        for result in partitionfunc(n - i, k - 1, i):
            yield (i,) + result


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
        for l in self.fc:
            x = l(x)
        return x


class Maron(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(Maron, self).__init__()

        #self.w = tf.Variable(tf.random.normal([84]), dtype=tf.float32, trainable=True)

        self.features = [
            tf.keras.layers.Dense(48, activation),
            # tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(6 * num_features, activation),
            tf.keras.layers.Dense(num_features, activation),
            # tf.keras.layers.Dense(1024, activation),
            # tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        self.a = list(set([p for x in partitionfunc(5, 5, l=0) for p in permutations(x)]))
        self.f = np.array(self.a)

    def call(self, x, training=None):
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
            p1, l1 = inv(a, b, c, d, e)
            p2, l2 = inv(e, a, b, c, d)
            p3, l3 = inv(d, e, a, b, c)
            p4, l4 = inv(c, d, e, a, b)
            p5, l5 = inv(b, c, d, e, a)
            q1 = p1 + p2 + p3 + p4 + p5
            L = (l1 + l2 + l3 + l4 + l5) / 5.
            return q1, L

        x, L = term()

        #x = tf.cast(tf.stack(terms, axis=1), tf.float32)
        for layer in self.features:
            x = layer(x)

        return x, L


class MessagePassing(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(MessagePassing, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, tf.keras.activations.tanh),
        ]
        self.M = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
        ]

        self.U = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(32, activation),
        ]

        self.R = [
            tf.keras.layers.Dense(32, activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, input, layers):
        for l in layers:
            input = l(input)
        return input

    def call(self, inputs, training=None):
        x = inputs[:, :, tf.newaxis]
        for layer in self.features:
            x = layer(x)
        a, b, c, d, e = tf.unstack(x, axis=1)

        Ua = a
        Ub = b
        Uc = c
        Ud = d
        Ue = e

        for i in range(1):
            Mab = self.process(tf.concat([Ua, Ub], axis=1), self.M)
            Mbc = self.process(tf.concat([Ub, Uc], axis=1), self.M)
            Mcd = self.process(tf.concat([Uc, Ud], axis=1), self.M)
            Mde = self.process(tf.concat([Ud, Ue], axis=1), self.M)
            Mea = self.process(tf.concat([Ue, Ua], axis=1), self.M)

            Ua = self.process(tf.concat([Mea, Ua, a], axis=1), self.U)
            Ub = self.process(tf.concat([Mab, Ub, b], axis=1), self.U)
            Uc = self.process(tf.concat([Mbc, Uc, c], axis=1), self.U)
            Ud = self.process(tf.concat([Mcd, Ud, d], axis=1), self.U)
            Ue = self.process(tf.concat([Mde, Ue, e], axis=1), self.U)

        x = self.process(tf.concat([Ua, Ub, Uc, Ud, Ue], axis=1), self.R)

        return x
