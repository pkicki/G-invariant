import inspect
import os
import sys
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath

import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()
tf.set_random_seed(444)
np.random.seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data[0], data[1])
            pbar.update(batch_size)


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

        self.m = 5
        n = 5
        p1 = np.eye(n)
        p = np.tile(np.eye(n)[np.newaxis], (self.m, 1, 1))

        #perm = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2], [4, 0, 1, 2, 3]] #Z5
        #perm = [[0, 1, 2, 3, 4], [1, 2, 3, 0, 4], [2, 3, 0, 1, 4], [3, 0, 1, 2, 4]] # Z4
        perm = [[0, 1, 2, 3, 4]] # Z4

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

        for layer in self.fc:
            x = layer(x)
        return x


def poly_Z5(x):
    def inv1(a, b):
        return a * b ** 2

    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = inv1(a, b) + inv1(b, c) + inv1(c, d) + inv1(d, e) + inv1(e, a)
    return q1


def main():
    # 1. Get datasets
    batch_size = 512
    ts = int(1e1)
    train_size = int(batch_size * ts)

    d = 5
    train_ds = np.random.rand(ts, batch_size, d)

    # 2. Define model
    n = 32
    # 3. Optimization

    # 5. Run everything
    a = []
    for k in range(30):
        model = GroupInvariance(n)
        acc = []
        for i in tqdm(range(ts), "Train"):
            pred = model(train_ds[i], training=True)
            y = poly_Z5(train_ds[i])
            model_loss = tf.keras.losses.mean_absolute_error(y[:, tf.newaxis], pred)
            acc = acc + list(model_loss.numpy())
        #print(np.mean(acc))
        a.append(np.mean(acc))
    print("FINAL:", np.mean(a))


if __name__ == '__main__':
    main()
