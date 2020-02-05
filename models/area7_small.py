from itertools import permutations
from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

tf.enable_eager_execution()


def groupAvereaging(inputs, operation):
    x = inputs
    x1 = x
    x2 = tf.roll(x, 1, 1)
    x3 = tf.roll(x, 2, 1)
    x4 = tf.roll(x, 3, 1)
    x5 = tf.roll(x, 4, 1)
    x6 = tf.roll(x, 5, 1)
    x7 = tf.roll(x, 6, 1)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)
    x5 = operation(x5)
    x6 = operation(x6)
    x7 = operation(x7)

    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4, x5, x6, x7], -1), -1)
    return x


class GroupInvariance(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvariance, self).__init__()
        self.features = [
            tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(7 * 2, None),
        ]

        self.fc = [
            tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, n_points, -1, 7))
        a, b, c, d, e, f, g = tf.unstack(x, axis=1)
        x = a[:, :, 0] * b[:, :, 1] * c[:, :, 2] * d[:, :, 3] * e[:, :, 4] * f[:, :, 5] * g[:, :, 6] \
            + b[:, :, 0] * c[:, :, 1] * d[:, :, 2] * e[:, :, 3] * f[:, :, 4] * g[:, :, 5] * a[:, :, 6]\
            + c[:, :, 0] * d[:, :, 1] * e[:, :, 2] * f[:, :, 3] * g[:, :, 4] * a[:, :, 5] * b[:, :, 6] \
            + d[:, :, 0] * e[:, :, 1] * f[:, :, 2] * g[:, :, 3] * a[:, :, 4] * b[:, :, 5] * c[:, :, 6]\
            + e[:, :, 0] * f[:, :, 1] * g[:, :, 2] * a[:, :, 3] * b[:, :, 4] * c[:, :, 5] * d[:, :, 6]\
            + f[:, :, 0] * g[:, :, 1] * a[:, :, 2] * b[:, :, 3] * c[:, :, 4] * d[:, :, 5] * e[:, :, 6]\
            + g[:, :, 0] * a[:, :, 1] * b[:, :, 2] * c[:, :, 3] * d[:, :, 4] * e[:, :, 5] * f[:, :, 6]

        for layer in self.fc:
            x = layer(x)

        return x


class GroupInvarianceConv(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(GroupInvarianceConv, self).__init__()

        activation = tf.keras.activations.tanh
        self.last_n = 2
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(7 * self.last_n, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)

        x = tf.reshape(x, (-1, 7, 7, self.last_n))
        a, b, c, d, e, f, g = tf.unstack(x, axis=1)

        x = a[:, 0] * b[:, 1] * c[:, 2] * d[:, 3] * e[:, 4] * f[:, 5] * g[:, 6] \
            + b[:, 0] * c[:, 1] * d[:, 2] * e[:, 3] * f[:, 4] * g[:, 5] * a[:, 6] \
            + c[:, 0] * d[:, 1] * e[:, 2] * f[:, 3] * g[:, 4] * a[:, 5] * b[:, 6] \
            + d[:, 0] * e[:, 1] * f[:, 2] * g[:, 3] * a[:, 4] * b[:, 5] * c[:, 6] \
            + e[:, 0] * f[:, 1] * g[:, 2] * a[:, 3] * b[:, 4] * c[:, 5] * d[:, 6] \
            + f[:, 0] * g[:, 1] * a[:, 2] * b[:, 3] * c[:, 4] * d[:, 5] * e[:, 6] \
            + g[:, 0] * a[:, 1] * b[:, 2] * c[:, 3] * d[:, 4] * e[:, 5] * f[:, 6]

        for layer in self.fc:
            x = layer(x)

        return x


class Conv1d(tf.keras.Model):
    def __init__(self, num_features):
        super(Conv1d, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 2  # 128
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
        #x = tf.concat([quad, quad[:, :1]], axis=1)
        x = tf.concat([quad[:, -1:], quad, quad[:, :1]], axis=1)
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


class SimpleNet(tf.keras.Model):
    def __init__(self, num_features):
        super(SimpleNet, self).__init__()
        activation = tf.keras.activations.tanh
        self.features = [
            #tf.keras.layers.Dense(128, activation),
            tf.keras.layers.Dense(64, activation),
            # tf.keras.layers.Dense(2048, activation),
            tf.keras.layers.Dense(20, activation),
            # tf.keras.layers.Dense(1024, activation),
            # tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, quad):
        x = tf.reshape(quad, (-1, 14))
        for layer in self.features:
            x = layer(x)

        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        # x = self.process(inputs)
        return x


class SegmentNet(tf.keras.Model):
    def __init__(self, num_features):
        super(SegmentNet, self).__init__()
        activation = tf.keras.activations.tanh
        self.features = [
            tf.keras.layers.Dense(64, activation),
        ]

        self.fc = [
            tf.keras.layers.Dense(28, activation),
            tf.keras.layers.Dense(1),
        ]

    def process(self, quad):
        s1 = tf.concat([quad[:, 0], quad[:, 1]], axis=-1)
        s2 = tf.concat([quad[:, 1], quad[:, 2]], axis=-1)
        s3 = tf.concat([quad[:, 2], quad[:, 3]], axis=-1)
        s4 = tf.concat([quad[:, 3], quad[:, 4]], axis=-1)
        s5 = tf.concat([quad[:, 4], quad[:, 5]], axis=-1)
        s6 = tf.concat([quad[:, 5], quad[:, 6]], axis=-1)
        s7 = tf.concat([quad[:, 6], quad[:, 0]], axis=-1)

        for layer in self.features:
            s1 = layer(s1)
            s2 = layer(s2)
            s3 = layer(s3)
            s4 = layer(s4)
            s5 = layer(s5)
            s6 = layer(s6)
            s7 = layer(s7)

        x = s1 + s2 + s3 + s4 + s5 + s6 + s7
        for layer in self.fc:
            x = layer(x)
        return x

    def call(self, inputs, training=None):
        # x = groupAvereaging(inputs, self.process)
        x = self.process(inputs)
        return x


class ConvImg(tf.keras.Model):
    def __init__(self, num_features):
        super(ConvImg, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 128  # * 3
        self.features = [
            # tf.keras.layers.Conv2D(16, 3, activation=activation),
            tf.keras.layers.Conv2D(4, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(4, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(4, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(8, 3, padding='same', activation=activation),
            tf.keras.layers.MaxPool2D((2, 2), padding='same'),
        ]
        # self.fc = tf.keras.layers.Dense(num_features, activation=activation)
        self.fc = [
            tf.keras.layers.Dense(16, activation=activation),
            tf.keras.layers.Dense(1),
        ]

        self.f = tf.keras.layers.Flatten()

    def process(self, quad):
        x = quad
        for layer in self.features:
            x = layer(x)
        x = self.f(x)

        for layer in self.fc:
            x = layer(x)

        return x

    def call(self, inputs, training=None):
        #x = groupAvereaging(inputs, self.process)
        x = self.process(inputs)
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
            #tf.keras.layers.Dense(128, activation),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x):
        for l in self.fc:
            x = l(x)
        return x


class Maron(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(Maron, self).__init__()

        self.features = [
            #tf.keras.layers.Dense(16, activation),
            #tf.keras.layers.Dense(64, activation),
            # tf.keras.layers.Dense(2048, activation),
            # tf.keras.layers.Dense(6 * num_features, activation),
            tf.keras.layers.Dense(16, activation),
            # tf.keras.layers.Dense(1024, activation),
            #tf.keras.layers.Dense(16, activation),
            tf.keras.layers.Dense(1),
        ]

        self.mulnn = MulNet()

        #self.a = list(set([p for x in partitionfunc(6, 12, l=0) for p in permutations(x)]))
        self.a = list(set([p for x in partitionfunc(7, 4, l=0) for p in permutations(x)]))
        self.a = [(x[:2] + (0, 0, 0, 0, 0) + x[2:] + (0, 0, 0, 0, 0)) for x in self.a]
        #self.a = []
        #for x in partitionfunc(7, 4, l=0):
        #    for p in permutations(x):
        #        print(p)
        #        self.a.append(p)
        self.f = np.array(self.a)

    def call(self, x, training=None):
        def inv(a, b, c, d, e, f, g, h, i, j, k, l, m, n):
            p = self.f
            x1 = a ** p[:, 0]
            x2 = b ** p[:, 1]
            x3 = c ** p[:, 2]
            x4 = d ** p[:, 3]
            x5 = e ** p[:, 4]
            x6 = f ** p[:, 5]
            x7 = g ** p[:, 6]
            x8 = h ** p[:, 7]
            x9 = i ** p[:, 8]
            x10 = j ** p[:, 9]
            x11 = k ** p[:, 10]
            x12 = l ** p[:, 11]
            x13 = m ** p[:, 12]
            x14 = n ** p[:, 13]
            mul = x1 * x2 * x3 * x4 * x5 * x6 * x7 * x8 * x9 * x10 * x11 * x12 * x13 * x14
            mulnn = self.mulnn(tf.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14], axis=-1))[:, :, 0]
            mul_loss = tf.keras.losses.mean_absolute_error(mul, mulnn)
            return mulnn, mul_loss

            # p = self.f
            # return a ** p[:, 0] * b ** p[:, 1] * c ** p[:, 2] * d ** p[:, 3] * e ** p[:, 4] * f ** p[:, 5] *\
            # g ** p[:, 6] * h ** p[:, 7]

        x = tf.transpose(x, (0, 2, 1))
        x = tf.reshape(x, (-1, 14))
        a, b, c, d, e, f, g, h, i, j, k, l, m, n = tf.unstack(x[:, :, tf.newaxis], axis=1)

        def term():
            p1, l1 = inv(a, b, c, d, e, f, g, h, i, j, k, l, m, n)
            p2, l2 = inv(g, a, b, c, d, e, f, n, h, i, j, k, l, m)
            p3, l3 = inv(f, g, a, b, c, d, e, m, n, h, i, j, k, l)
            p4, l4 = inv(e, f, g, a, b, c, d, l, m, n, h, i, j, k)
            p5, l5 = inv(d, e, f, g, a, b, c, k, l, m, n, h, i, j)
            p6, l6 = inv(c, d, e, f, g, a, b, j, k, l, m, n, h, i)
            p7, l7 = inv(b, c, d, e, f, g, a, i, j, k, l, m, n, h)

            q1 = p1 + p2 + p3 + p4 + p5 + p6 + p7
            L = l1 + l2 + l3 + l4 + l5 + l6 + l7
            return q1, L

        x, L = term()

        for layer in self.features:
            x = layer(x)

        return x, L


class MessagePassing(tf.keras.Model):
    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(MessagePassing, self).__init__()
        n = 20
        self.features = [
            tf.keras.layers.Dense(n, tf.keras.activations.tanh),
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
        #x = inputs[:, :, tf.newaxis]
        x = inputs
        for layer in self.features:
            x = layer(x)
        a, b, c, d, e, f, g = tf.unstack(x, axis=1)

        Ua = a
        Ub = b
        Uc = c
        Ud = d
        Ue = e
        Uf = f
        Ug = g

        for i in range(1):
            Mab = self.process(tf.concat([Ua, Ub], axis=1), self.M)
            Mbc = self.process(tf.concat([Ub, Uc], axis=1), self.M)
            Mcd = self.process(tf.concat([Uc, Ud], axis=1), self.M)
            Mde = self.process(tf.concat([Ud, Ue], axis=1), self.M)
            Mef = self.process(tf.concat([Ue, Uf], axis=1), self.M)
            Mfg = self.process(tf.concat([Uf, Ug], axis=1), self.M)
            Mga = self.process(tf.concat([Ug, Ua], axis=1), self.M)

            Ua = self.process(tf.concat([Mga, Ua, a], axis=1), self.U)
            Ub = self.process(tf.concat([Mab, Ub, b], axis=1), self.U)
            Uc = self.process(tf.concat([Mbc, Uc, c], axis=1), self.U)
            Ud = self.process(tf.concat([Mcd, Ud, d], axis=1), self.U)
            Ue = self.process(tf.concat([Mde, Ue, e], axis=1), self.U)
            Uf = self.process(tf.concat([Mef, Uf, f], axis=1), self.U)
            Ug = self.process(tf.concat([Mfg, Ug, g], axis=1), self.U)

        x = self.process(tf.concat([Ua, Ub, Uc, Ud, Ue, Uf, Ug], axis=1), self.R)

        return x

def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space, _ = data

    for i in range(free_space.shape[1]):
        for j in range(3):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    plt.xlim(-25.0, 25.0)
    plt.ylim(0.0, 50.0)
    # plt.xlim(-15.0, 20.0)
    # plt.ylim(0.0, 35.0)
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()
