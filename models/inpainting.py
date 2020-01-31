from math import pi

import tensorflow as tf
from matplotlib import pyplot as plt

tf.enable_eager_execution()


class InpaintingNet(tf.keras.Model):
    def __init__(self):
        super(InpaintingNet, self).__init__()
        self.n = 64
        self.quad_processor = GroupInvariance(self.n)
        #self.quad_processor = GroupInvarianceConv(self.n)
        #self.quad_processor = SimpleNet(self.n)
        #self.quad_processor = Conv1d(self.n)
        self.decoder = Decoder()

    def call(self, quad):
        ft = self.quad_processor(quad)
        ft = tf.reshape(ft, (-1, 2, 2, 16))
        img = self.decoder(ft)
        img = tf.squeeze(img)

        return img


class Decoder(tf.keras.Model):
    def __init__(self, activation=tf.keras.activations.tanh):
        super(Decoder, self).__init__()
        self.features = [
            tf.keras.layers.Conv2DTranspose(256, 3, 2, padding='same', activation=activation),
            tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same', activation=activation),
            tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same', activation=activation),
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation=activation),
            tf.keras.layers.Conv2DTranspose(1, 3, 2, padding='same', activation=tf.keras.activations.sigmoid),
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
        return x


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
            # tf.keras.layers.Dense(16, activation),
            # tf.keras.layers.Dense(64, activation),
            # tf.keras.layers.Dense(4 * 64, tf.keras.activations.tanh),
            tf.keras.layers.Dense(4 * 168, tf.keras.activations.tanh),
        ]

        self.fc = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(num_features, activation),
        ]

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        n_points = x.shape[1]
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (bs, n_points, -1, 4))
        a, b, c, d = tf.unstack(x, axis=1)
        # x = a[:, :, 0] + b[:, :, 1] + c[:, :, 2] + d[:, :, 3] \
        #    + b[:, :, 0] + c[:, :, 1] + d[:, :, 2] + a[:, :, 3] \
        #    + c[:, :, 0] + d[:, :, 1] + a[:, :, 2] + b[:, :, 3] \
        #    + d[:, :, 0] + a[:, :, 1] + b[:, :, 2] + c[:, :, 3]

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
        self.last_n = 88
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            tf.keras.layers.Conv1D(4 * self.last_n, 1, padding='same'),
        ]
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
        ]

    def call(self, inputs, training=None):
        x = tf.concat([inputs[:, -1:], inputs, inputs[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)
        # x =
        a, b, c, d = tf.unstack(x, axis=1)
        a = tf.reshape(a, (-1, 4, self.last_n))
        b = tf.reshape(b, (-1, 4, self.last_n))
        c = tf.reshape(c, (-1, 4, self.last_n))
        d = tf.reshape(d, (-1, 4, self.last_n))

        x = a[:, 0] * b[:, 1] * c[:, 2] * d[:, 3] \
            + b[:, 0] * c[:, 1] * d[:, 2] * a[:, 3] \
            + c[:, 0] * d[:, 1] * a[:, 2] * b[:, 3] \
            + d[:, 0] * a[:, 1] * b[:, 2] * c[:, 3]

        for layer in self.fc:
            x = layer(x)

        return x


class Conv1d(tf.keras.Model):
    def __init__(self, num_features):
        super(Conv1d, self).__init__()
        activation = tf.keras.activations.tanh
        self.last_n = 112  # * 4
        self.features = [
            tf.keras.layers.Conv1D(32, 3, activation=activation),
            # tf.keras.layers.Conv1D(64, 3, activation=activation),
            # tf.keras.layers.Conv1D(self.last_n, 1, padding='same', activation=activation),
            tf.keras.layers.Conv1D(self.last_n, 3, activation=activation),
        ]
        # self.fc = tf.keras.layers.Dense(num_features, activation=activation)
        self.fc = [
            tf.keras.layers.Dense(32, activation=activation),
            tf.keras.layers.Dense(num_features, activation=activation),
        ]

    def process(self, quad):
        # x = tf.reshape(quad, (-1, 8))
        x = tf.concat([quad, quad[:, :1]], axis=1)
        for layer in self.features:
            x = layer(x)
        x = tf.reshape(x, (-1, self.last_n))
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
        # self.features = [
        #    tf.keras.layers.Dense(64, activation),
        #    tf.keras.layers.Dense(6 * num_features, activation),
        #    tf.keras.layers.Dense(num_features, activation),
        # ]
        self.features = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(int(1.5 * num_features), activation),
            tf.keras.layers.Dense(num_features, activation),
        ]

    def process(self, quad):
        x = tf.reshape(quad, (-1, 8))
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
            tf.keras.layers.Dense(6 * num_features, activation),
        ]

        self.fc = tf.keras.layers.Dense(num_features, activation)

    def process(self, quad):
        s1 = tf.concat([quad[:, 0], quad[:, 1]], axis=-1)
        s2 = tf.concat([quad[:, 1], quad[:, 2]], axis=-1)
        s3 = tf.concat([quad[:, 2], quad[:, 3]], axis=-1)
        s4 = tf.concat([quad[:, 3], quad[:, 0]], axis=-1)

        for layer in self.features:
            s1 = layer(s1)
            s2 = layer(s2)
            s3 = layer(s3)
            s4 = layer(s4)

        x = s1 + s2 + s3 + s4
        x = self.fc(x)
        return x

    def call(self, inputs, training=None):
        x = groupAvereaging(inputs, self.process)
        # x = self.process(inputs)
        return x


def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space, _ = data

    for i in range(free_space.shape[1]):
        for j in range(4):
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
