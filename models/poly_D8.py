from itertools import permutations
from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

tf.enable_eager_execution()


def groupAvereaging(inputs, operation):
    x = inputs
    a, b, c, d, e = tf.unstack(x, axis=1)

    # D8 in S5
    x1 = x
    x2 = tf.stack([b, c, d, a, e], axis=1)
    x3 = tf.stack([c, d, a, b, e], axis=1)
    x4 = tf.stack([d, a, b, c, e], axis=1)

    x5 = tf.stack([d, c, b, a, e], axis=1)
    x6 = tf.stack([c, b, a, d, e], axis=1)
    x7 = tf.stack([b, a, d, c, e], axis=1)
    x8 = tf.stack([a, d, c, b, e], axis=1)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)
    x5 = operation(x5)
    x6 = operation(x6)
    x7 = operation(x7)
    x8 = operation(x8)

    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4, x5, x6, x7, x8], -1), -1)
    return x


class GroupInvariance(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvariance, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(5 * 2, tf.keras.activations.sigmoid),
            # tf.keras.layers.Dense(5 * 64),
        ]

        self.fc = [
            # tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(num_features, tf.keras.activations.relu, use_bias=False),
            tf.keras.layers.Dense(1),
        ]

        self.m = 8
        n = 5
        p1 = np.eye(n)
        p = np.tile(np.eye(n)[np.newaxis], (self.m, 1, 1))

        perm = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2],
                [3, 2, 1, 0], [2, 1, 0, 3], [1, 0, 3, 2], [0, 3, 2, 1]]

        perm = [list(x) + [4] for x in perm]

        for i, x in enumerate(perm):
            p[i, x, :] = p1[np.arange(n)]

        self.p = p

    def call(self, inputs, training=None):
        x = inputs[:, :, tf.newaxis]
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, n_points, -1, 5))

        fin = x
        fin = np.transpose(fin, (0, 2, 1, 3))
        fin = fin[:, :, np.newaxis]
        fin = np.tile(fin, (1, 1, self.m, 1, 1))
        y = fin @ self.p
        y = y[:, :, :, np.arange(5), np.arange(5)]
        y = np.prod(y, axis=3)
        y = np.sum(y, axis=2)
        x = y

        #a, b, c, d, e = tf.unstack(x, axis=1)

        # D8 in S5
        #x = a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
        #    + d[:, :, 0] * a[:, :, 1] * b[:, :, 2] * c[:, :, 3] * e[:, :, 4] \
        #    + c[:, :, 0] * d[:, :, 1] * a[:, :, 2] * b[:, :, 3] * e[:, :, 4] \
        #    + b[:, :, 0] * c[:, :, 1] * d[:, :, 2] * a[:, :, 3] * e[:, :, 4] \
        #    + d[:, :, 0] * c[:, :, 1] * b[:, :, 2] * a[:, :, 3] * e[:, :, 4] \
        #    + c[:, :, 0] * b[:, :, 1] * a[:, :, 2] * d[:, :, 3] * e[:, :, 4] \
        #    + b[:, :, 0] * a[:, :, 1] * d[:, :, 2] * c[:, :, 3] * e[:, :, 4] \
        #    + a[:, :, 0] * d[:, :, 1] * c[:, :, 2] * b[:, :, 3] * e[:, :, 4]

        for layer in self.fc:
            x = layer(x)
        x = tf.reduce_sum(x, 1, keep_dims=True)

        return x


class SimpleNet(tf.keras.Model):
    def __init__(self, num_features):
        super(SimpleNet, self).__init__()
        activation = tf.keras.activations.tanh
        self.features = [
            tf.keras.layers.Dense(48, activation),
            # tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(32, activation),
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
        self.last_n = 2#116#128
        self.conv = tf.keras.layers.Conv1D(32, 3, activation=activation)
        self.e = tf.keras.layers.Dense(32, activation=activation)
        self.features = [
            #tf.keras.layers.Conv1D(32, 3, activation=activation),
            #tf.keras.layers.Conv1D(5 * self.last_n, 1, padding='same'),
            tf.keras.layers.Dense(5 * self.last_n),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = inputs
        inputs = x[:, :-1]
        e = x[:, -1:, tf.newaxis]
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)[:, :, tf.newaxis]
        bs = x.shape[0]
        x = self.conv(x)
        e = self.e(e)
        x = tf.concat([x, e], axis=1)
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, 5, 5, self.last_n))
        a, b, c, d, e = tf.unstack(x, axis=1)

        # D8 in S5
        x = a[:, 0] * b[:, 1] * c[:, 2] * d[:, 3] * e[:, 4] \
            + d[:, 0] * a[:, 1] * b[:, 2] * c[:, 3] * e[:, 4] \
            + c[:, 0] * d[:, 1] * a[:, 2] * b[:, 3] * e[:, 4] \
            + b[:, 0] * c[:, 1] * d[:, 2] * a[:, 3] * e[:, 4] \
            + d[:, 0] * c[:, 1] * b[:, 2] * a[:, 3] * e[:, 4] \
            + c[:, 0] * b[:, 1] * a[:, 2] * d[:, 3] * e[:, 4] \
            + b[:, 0] * a[:, 1] * d[:, 2] * c[:, 3] * e[:, 4] \
            + a[:, 0] * d[:, 1] * c[:, 2] * b[:, 3] * e[:, 4]

        for layer in self.fc:
            x = layer(x)

        return x


class Conv1d(tf.keras.Model):
    def __init__(self, num_features):
        super(Conv1d, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 3  # 128
        self.conv = tf.keras.layers.Conv1D(32, 3, activation=activation)
        self.e = tf.keras.layers.Dense(32, activation=activation)
        self.features = [
            tf.keras.layers.Dense(self.last_n),
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
        quad = x[:, :-1]
        e = x[:, -1:]
        x = tf.concat([quad[:, -1:], quad, quad[:, :1]], axis=1)[:, :, tf.newaxis]
        x = self.conv(x)
        e = self.e(e[:, :, tf.newaxis])
        x = tf.concat([x, e], axis=1)
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
            tf.keras.layers.Dense(16, activation),
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
            tf.keras.layers.Dense(4, activation),
            #tf.keras.layers.Dense(48, activation),
            # tf.keras.layers.Dense(2048, activation),
            #tf.keras.layers.Dense(6 * num_features, activation),
            # tf.keras.layers.Dense(1024, activation),
            # tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        self.a = list(set([p for x in partitionfunc(8, 5, l=0) for p in permutations(x)]))
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
            p2, l2 = inv(d, a, b, c, e)
            p3, l3 = inv(c, d, a, b, e)
            p4, l4 = inv(b, c, d, a, e)
            p5, l5 = inv(d, c, b, a, e)
            p6, l6 = inv(a, d, c, b, e)
            p7, l7 = inv(b, a, d, c, e)
            p8, l8 = inv(c, b, a, d, e)
            q1 = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8
            L = (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8) / 8.
            return q1, L

        x, L = term()

        #x = tf.cast(tf.stack(terms, axis=1), tf.float32)
        for layer in self.features:
            x = layer(x)

        return x, L


class MessagePassing(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(MessagePassing, self).__init__()
        n = 17
        self.features = [
            tf.keras.layers.Dense(n, activation),
        ]
        self.M = [
            tf.keras.layers.Dense(n, activation),
        ]

        self.U = [
            tf.keras.layers.Dense(n, activation),
        ]

        self.R = [
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
            Mba = self.process(tf.concat([Ub, Ua], axis=1), self.M)
            Mbc = self.process(tf.concat([Ub, Uc], axis=1), self.M)
            Mcb = self.process(tf.concat([Uc, Ub], axis=1), self.M)
            Mcd = self.process(tf.concat([Uc, Ud], axis=1), self.M)
            Mdc = self.process(tf.concat([Ud, Uc], axis=1), self.M)
            Mda = self.process(tf.concat([Ud, Ua], axis=1), self.M)
            Mad = self.process(tf.concat([Ua, Ud], axis=1), self.M)

            Ua = self.process(tf.concat([Mab, Mad, Ua, a], axis=1), self.U)
            Ub = self.process(tf.concat([Mba, Mbc, Ub, b], axis=1), self.U)
            Uc = self.process(tf.concat([Mcb, Mcd, Uc, c], axis=1), self.U)
            Ud = self.process(tf.concat([Mdc, Mda, Ud, d], axis=1), self.U)
            Ue = self.process(tf.concat([tf.zeros_like(Mab), tf.zeros_like(Mab), Ue, e], axis=1), self.U)

        x = self.process(tf.concat([Ua, Ub, Uc, Ud, Ue], axis=1), self.R)

        return x
