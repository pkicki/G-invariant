import os
import numpy as np
from scipy.io import loadmat
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


def action_recognition_dataset(path):
    x = np.load(path + "x.npy", allow_pickle=True)
    #W = 500.
    #H = 300.
    #WH = np.array([W, H])
    #x /= WH
    x -= x[:, 7, tf.newaxis]
    y = np.load(path + "y.npy")
    cat = {cls: idx for idx, cls in enumerate(sorted(list(set(list(y)))))}
    n_cls = len(cat)
    y = np.array([cat[k] for k in list(y)])
    u, cnts = np.unique(y, return_counts=True)
    a = dict(zip(u, cnts))
    split = np.load(path + "split.npy")
    s1 = int(0.7 * len(x))
    s2 = int(0.9 * len(x))
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    idx_train = idx[:s1]
    idx_val = idx[s1:s2]
    idx_test = idx[s2:]
    x_train = x[idx_train]
    m = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True)
    x_train = (x_train - m) / (std + 1e-10)
    x_val = x[idx_val]
    x_val = (x_val - m) / (std + 1e-10)
    x_test = x[idx_test]
    x_test = (x_test - m) / (std + 1e-10)
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(buffer_size=len(x_train), reshuffle_each_iteration=True)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)) \
        .shuffle(buffer_size=len(x_val), reshuffle_each_iteration=True)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .shuffle(buffer_size=len(x_test), reshuffle_each_iteration=True)

    return train_ds, len(x_train), val_ds, len(x_val), test_ds, len(x_test), n_cls
