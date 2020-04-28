import os
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def quadrangle_area_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:4]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        area = data[8]
        return xy, area

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def H3O_dataset(path):
    xyzs = [[], [], [], []]
    e = []
    with open(path, 'r') as fh:
        lines = fh.read().split("\n")[:-1]
        for i, line in enumerate(lines):
            if i % 6 == 1:
                e.append(float(line))
            if i % 6 > 1:
                xyz = line.split()[1:]
                xyz = [float(x) for x in xyz]
                print(xyz)
                xyzs[(i % 6) - 2].append(xyz)
    xyzs = np.array(xyzs).transpose((1, 0, 2))
    rh1h2 = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 1]) ** 2, -1))
    rh1h3 = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 2]) ** 2, -1))
    rh2h3 = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 2]) ** 2, -1))
    rh1o = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 3]) ** 2, -1))
    rh2o = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 3]) ** 2, -1))
    rh3o = tf.sqrt(tf.reduce_sum((xyzs[:, 2] - xyzs[:, 3]) ** 2, -1))

    xyzs = tf.stack([rh1o, rh2o, rh3o, rh1h2, rh1h3, rh2h3], -1)[:, :, tf.newaxis]
    a = tf.exp(-xyzs / (2.5 * 0.5292))
    b = 1 / xyzs
    c = xyzs
    xyzs = tf.concat([a, b, c], -1)
    e = np.array(e, dtype=np.float32)[:, np.newaxis]

    xyzs = xyzs.numpy()
    size = xyzs.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    train_x = xyzs[idx[:int(0.7 * size)]]
    val_x = xyzs[idx[int(0.7 * size):int(0.9 * size)]]
    test_x = xyzs[idx[int(0.9 * size):]]
    train_y = e[idx[:int(0.7 * size)]]
    val_y = e[idx[int(0.7 * size):int(0.9 * size)]]
    test_y = e[idx[int(0.9 * size):]]

    m = tf.reduce_mean(train_x, 0, keepdims=True)
    std = tf.math.reduce_std(train_x, 0, keepdims=True)
    train_x = (train_x - m) / (std)
    val_x = (val_x - m) / (std)

    m = np.mean(train_y, 0, keepdims=True)
    std = np.std(train_y, 0, keepdims=True)
    train_y = (train_y - m) / (std)
    val_y = (val_y - m) / (std)
    test_y = (test_y - m) / (std)

    train_size = int(train_x.shape[0])
    val_size = int(val_x.shape[0])
    test_size = int(test_x.shape[0])

    def train_gen():
        for i in range(train_size):
            yield train_x[i], train_y[i]

    def val_gen():
        for i in range(val_size):
            yield val_x[i], val_y[i]

    def test_gen():
        for i in range(test_size):
            yield test_x[i], test_y[i]

    train_ds = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=train_size, reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_generator(val_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=val_size, reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=test_size, reshuffle_each_iteration=True)

    return train_ds, train_size, val_ds, val_size, test_ds, test_size, m, std


def H2CO_pos_dataset(path):
    xyzs = [[], [], [], []]
    e = []
    with open(path, 'r') as fh:
        lines = fh.read().split("\n")[:-1]
        for i, line in enumerate(lines):
            if i % 6 == 1:
                e.append(float(line))
            if i % 6 > 1:
                xyz = line.split()[1:]
                xyz = [float(x) for x in xyz]
                #print(xyz)
                xyzs[(i % 6) - 2].append(xyz)
                if (i % 6) - 2 == 0 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 6) - 2 == 1 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 6) - 2 == 2 and line.split()[0] != "C":
                    print("ZUO")
                if (i % 6) - 2 == 3 and line.split()[0] != "O":
                    print("ZUO")
    xyzs = np.array(xyzs).transpose((1, 0, 2))
    xyzs_m = np.mean(xyzs, axis=1, keepdims=True)
    xyzs = xyzs - xyzs_m
    cov = (np.transpose(xyzs, (0, 2, 1)) @ xyzs) / 3.
    _, v = np.linalg.eigh(cov)
    f = xyzs @ v

    e = np.array(e, dtype=np.float32)[:, np.newaxis]
    size = xyzs.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    train_x = xyzs[idx[:int(0.7 * size)]]
    val_x = xyzs[idx[int(0.7 * size):int(0.9 * size)]]
    test_x = xyzs[idx[int(0.9 * size):]]
    train_y = e[idx[:int(0.7 * size)]]
    val_y = e[idx[int(0.7 * size):int(0.9 * size)]]
    test_y = e[idx[int(0.9 * size):]]

    m = tf.reduce_mean(train_x, 0, keepdims=True)
    std = tf.math.reduce_std(train_x, 0, keepdims=True)
    train_x = (train_x - m) / (std)
    val_x = (val_x - m) / (std)

    m = np.mean(train_y, 0, keepdims=True)
    std = np.std(train_y, 0, keepdims=True)
    train_y = (train_y - m) / (std)
    val_y = (val_y - m) / (std)
    test_y = (test_y - m) / (std)

    train_size = int(train_x.shape[0])
    val_size = int(val_x.shape[0])
    test_size = int(test_x.shape[0])

    def train_gen():
        for i in range(train_size):
            yield train_x[i], train_y[i]

    def val_gen():
        for i in range(val_size):
            yield val_x[i], val_y[i]

    def test_gen():
        for i in range(test_size):
            yield test_x[i], test_y[i]

    train_ds = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=train_size, reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_generator(val_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=val_size, reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=test_size, reshuffle_each_iteration=True)

    return train_ds, train_size, val_ds, val_size, test_ds, test_size, m, std

def H2CO_dataset(path):
    xyzs = [[], [], [], []]
    e = []
    with open(path, 'r') as fh:
        lines = fh.read().split("\n")[:-1]
        for i, line in enumerate(lines):
            if i % 6 == 1:
                e.append(float(line))
            if i % 6 > 1:
                xyz = line.split()[1:]
                xyz = [float(x) for x in xyz]
                #print(xyz)
                xyzs[(i % 6) - 2].append(xyz)
                if (i % 6) - 2 == 0 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 6) - 2 == 1 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 6) - 2 == 2 and line.split()[0] != "C":
                    print("ZUO")
                if (i % 6) - 2 == 3 and line.split()[0] != "O":
                    print("ZUO")
    xyzs = np.array(xyzs).transpose((1, 0, 2))
    rh1h2 = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 1]) ** 2, -1))
    rh1c = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 2]) ** 2, -1))
    rh2c = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 2]) ** 2, -1))
    rh1o = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 3]) ** 2, -1))
    rh2o = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 3]) ** 2, -1))
    rco = tf.sqrt(tf.reduce_sum((xyzs[:, 2] - xyzs[:, 3]) ** 2, -1))

    xyzs = tf.stack([rh1h2, rh1c, rh2c, rh1o, rh2o, rco], -1)[:, :, tf.newaxis]
    a = tf.exp(-xyzs / (2.5 * 0.5292))
    b = 1 / xyzs
    c = xyzs
    xyzs = tf.concat([a, b, c], -1)
    e = np.array(e, dtype=np.float32)[:, np.newaxis]

    xyzs = xyzs.numpy()
    size = xyzs.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    train_x = xyzs[idx[:int(0.7 * size)]]
    val_x = xyzs[idx[int(0.7 * size):int(0.9 * size)]]
    test_x = xyzs[idx[int(0.9 * size):]]
    train_y = e[idx[:int(0.7 * size)]]
    val_y = e[idx[int(0.7 * size):int(0.9 * size)]]
    test_y = e[idx[int(0.9 * size):]]

    m = tf.reduce_mean(train_x, 0, keepdims=True)
    std = tf.math.reduce_std(train_x, 0, keepdims=True)
    train_x = (train_x - m) / (std)
    val_x = (val_x - m) / (std)

    m = np.mean(train_y, 0, keepdims=True)
    std = np.std(train_y, 0, keepdims=True)
    train_y = (train_y - m) / (std)
    val_y = (val_y - m) / (std)
    test_y = (test_y - m) / (std)

    train_size = int(train_x.shape[0])
    val_size = int(val_x.shape[0])
    test_size = int(test_x.shape[0])

    def train_gen():
        for i in range(train_size):
            yield train_x[i], train_y[i]

    def val_gen():
        for i in range(val_size):
            yield val_x[i], val_y[i]

    def test_gen():
        for i in range(test_size):
            yield test_x[i], test_y[i]

    train_ds = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=train_size, reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_generator(val_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=val_size, reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=test_size, reshuffle_each_iteration=True)

    return train_ds, train_size, val_ds, val_size, test_ds, test_size, m, std


def HCOOH2_dataset(path):
    """TODO NOT FINISHED"""
    xyzs = [[], [], [], []]
    e = []
    with open(path, 'r') as fh:
        lines = fh.read().split("\n")[:-1]
        for i, line in enumerate(lines):
            if i % 6 == 1:
                e.append(float(line.split()[1]))
            if i % 6 > 1:
                xyz = line.split()[1:]
                xyz = [float(x) for x in xyz]
                #print(xyz)
                xyzs[(i % 12) - 2].append(xyz)
                if (i % 12) - 2 == 0 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 12) - 2 == 1 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 12) - 2 == 2 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 12) - 2 == 3 and line.split()[0] != "H":
                    print("ZUO")
                if (i % 12) - 2 == 4 and line.split()[0] != "O":
                    print("ZUO")
                if (i % 12) - 2 == 5 and line.split()[0] != "O":
                    print("ZUO")
                if (i % 12) - 2 == 6 and line.split()[0] != "O":
                    print("ZUO")
                if (i % 12) - 2 == 7 and line.split()[0] != "O":
                    print("ZUO")
                if (i % 12) - 2 == 8 and line.split()[0] != "C":
                    print("ZUO")
                if (i % 12) - 2 == 9 and line.split()[0] != "C":
                    print("ZUO")
    xyzs = np.array(xyzs).transpose((1, 0, 2))
    rh1h2 = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 1]) ** 2, -1))
    rh1c = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 2]) ** 2, -1))
    rh2c = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 2]) ** 2, -1))
    rh1o = tf.sqrt(tf.reduce_sum((xyzs[:, 0] - xyzs[:, 3]) ** 2, -1))
    rh2o = tf.sqrt(tf.reduce_sum((xyzs[:, 1] - xyzs[:, 3]) ** 2, -1))
    rco = tf.sqrt(tf.reduce_sum((xyzs[:, 2] - xyzs[:, 3]) ** 2, -1))

    xyzs = tf.stack([rh1h2, rh1c, rh2c, rh1o, rh2o, rco], -1)[:, :, tf.newaxis]
    a = tf.exp(-xyzs / (2.5 * 0.5292))
    b = 1 / xyzs
    c = xyzs
    xyzs = tf.concat([a, b, c], -1)
    e = np.array(e, dtype=np.float32)[:, np.newaxis]

    xyzs = xyzs.numpy()
    size = xyzs.shape[0]
    idx = np.arange(size)
    np.random.shuffle(idx)
    train_x = xyzs[idx[:int(0.7 * size)]]
    val_x = xyzs[idx[int(0.7 * size):int(0.9 * size)]]
    test_x = xyzs[idx[int(0.9 * size):]]
    train_y = e[idx[:int(0.7 * size)]]
    val_y = e[idx[int(0.7 * size):int(0.9 * size)]]
    test_y = e[idx[int(0.9 * size):]]

    m = tf.reduce_mean(train_x, 0, keepdims=True)
    std = tf.math.reduce_std(train_x, 0, keepdims=True)
    train_x = (train_x - m) / (std)
    val_x = (val_x - m) / (std)

    m = np.mean(train_y, 0, keepdims=True)
    std = np.std(train_y, 0, keepdims=True)
    train_y = (train_y - m) / (std)
    val_y = (val_y - m) / (std)
    test_y = (test_y - m) / (std)

    train_size = int(train_x.shape[0])
    val_size = int(val_x.shape[0])
    test_size = int(test_x.shape[0])

    def train_gen():
        for i in range(train_size):
            yield train_x[i], train_y[i]

    def val_gen():
        for i in range(val_size):
            yield val_x[i], val_y[i]

    def test_gen():
        for i in range(test_size):
            yield test_x[i], test_y[i]

    train_ds = tf.data.Dataset.from_generator(train_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=train_size, reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_generator(val_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=val_size, reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.from_generator(test_gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=test_size, reshuffle_each_iteration=True)

    return train_ds, train_size, val_ds, val_size, test_ds, test_size, m, std
