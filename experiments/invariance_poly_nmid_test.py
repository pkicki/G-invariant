import inspect
import os
import sys
from glob import glob
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from dataset.scenarios import area4_dataset
from models.poly_nmid import GroupInvariance, GroupInvarianceConv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import LoadFromFile

tf.enable_eager_execution()
#tf.set_random_seed(444)
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


def poly_Z5(x):
    def inv1(a, b):
        return a * b ** 2

    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = inv1(a, b) + inv1(b, c) + inv1(c, d) + inv1(d, e) + inv1(e, a)
    return q1


names = ["my_inv_conv", "my_inv_fc"]
#names = ["my_inv_conv"]
#names = ["my_inv_fc"]

def secondary():
    batch_size = 16

    ts = int(1e0)
    vs = int(3e1)
    s = int(3e2)

    d = 5
    train_ds = np.random.rand(ts, batch_size, d)
    val_ds = np.random.rand(vs, batch_size, d)
    test_ds = np.random.rand(s, batch_size, d)

    datasets = [("train", train_ds, ts), ("val", val_ds, vs), ("test", test_ds, s)]
    #datasets = [("train", train_ds, ts), ("val", val_ds, vs)]
    #datasets = [("train", train_ds, ts)]

    results = []
    for name in names:
        #for i in [8]:
        for i in [1, 2, 4, 8, 16, 32, 64, 128]:
            if "conv" in name:
                model = GroupInvarianceConv(i)
            else:
                model = GroupInvariance(i)
            fname = name + "_" + str(i)
            print(fname)
            for ds_type, ds, ds_size in datasets:
                mae = []
                times = []
                for k in range(1, 10):
                    p = glob("./working_dir/poly_nmid/" + fname + "_" + str(k) + "/checkpoints/best*.index")
                    best_path = sorted(p, key=lambda x: (len(x), x))[-1].replace(".index", "")
                    print(best_path)
                    #model.load_weights(best_path).expect_partial()
                    model.load_weights(best_path)

                    acc = []
                    for j in tqdm(range(ds_size)):

                        start = time()
                        pred = model(ds[j], training=False)
                        stop = time()
                        times.append(stop - start)

                        y = poly_Z5(ds[j])
                        model_loss = tf.keras.losses.mean_absolute_error(y[:, tf.newaxis], pred)
                        acc = acc + list(model_loss.numpy())

                    print(np.mean(acc))
                    mae.append(np.mean(acc))

                print(np.mean(mae))
                print(np.std(mae))
                r = (fname, ds_type, np.mean(mae), np.std(mae), np.mean(times[1:]), np.std(times[1:]))
                with open("./paper/poly_nmid.csv", 'a') as fh:
                    fh.write("%s\t%s\t%.5f\t%.5f\t%.6f\t%.6f\n" % r)


if __name__ == '__main__':
    secondary()
