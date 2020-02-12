import inspect
import os
import sys
from glob import glob
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from models.area import GroupInvariance, SimpleNet, Conv1d, GroupInvarianceConv, Maron
from utils.permutation_groups import Z4

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

tf.enable_eager_execution()
tf.set_random_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data[0], data[1])
            pbar.update(batch_size)


names = ["avg_conv", "avg_fc", "my_inv_conv", "my_inv_fc", "maron"]
#names = ["maron"]
#models = [Maron()]
models = [Conv1d(), SimpleNet(), GroupInvarianceConv(Z4, 2), GroupInvariance(Z4, 2), Maron()]


def secondary():
    batch_size = 64
    results = []

    for i, name in enumerate(names):
        print(name)
        for ds_type in ["train", "val", "test"]:
            scenario_path = "../../data_inv/train/area4paper_shift"
            ds, ds_size = scenarios.quadrangle_area_dataset(scenario_path.replace("train", ds_type))
            model = models[i]
            path = "./paper/area4"
            mae = []
            times = []
            for k in range(1, 10):
                best_path = sorted(glob(path + "/" + name + "_" + str(k) + "/checkpoints/best*.index"),
                                   key=lambda x: (len(x), x))[-1].replace(".index", "")
                model.load_weights(best_path).expect_partial()
                dataset_epoch = ds.shuffle(ds_size)
                dataset_epoch = dataset_epoch.batch(batch_size).prefetch(batch_size)

                acc = []
                for l, quad, area, in _ds('Train', dataset_epoch, ds_size, 0, batch_size):
                    # pred, L = model(quad, training=True)
                    start = time()
                    pred = model(quad)
                    end = time()
                    if len(pred) == 2:
                        pred = pred[0]

                    model_loss = tf.keras.losses.mean_absolute_error(area[:, tf.newaxis], pred)
                    # model_loss = model_loss / area
                    acc = acc + list(model_loss.numpy())
                    times.append(end - start)

                print(np.mean(acc))
                mae.append(np.mean(acc))

            print(np.mean(mae))
            print(np.std(mae))
            results.append((name, ds_type, np.mean(mae), np.std(mae), np.mean(times), np.std(times)))
    with open("./paper/area4.csv", 'w') as fh:
        for r in results:
            fh.write("%s\t%s\t%.5f\t%.5f\t%.6f\t%.6f\n" % r)


if __name__ == '__main__':
    secondary()
