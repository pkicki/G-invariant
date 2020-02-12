import inspect
import os
import sys
from glob import glob
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from experiments.invariance_poly_Z5 import poly_Z5
from models.poly import *
from utils.permutation_groups import Z5

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath

import tensorflow as tf
from tqdm import tqdm

tf.enable_eager_execution()
# tf.set_random_seed(444)
np.random.seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def main():
    batch_size = 16
    # 1. Get datasets
    ts = int(1e0)
    vs = int(3e1)
    s = int(3e2)

    d = 5
    train_ds = np.random.rand(ts, batch_size, d)
    val_ds = np.random.rand(vs, batch_size, d)
    test_ds = np.random.rand(s, batch_size, d)

    #dss = [("train", train_ds)]
    dss = [("train", train_ds), ("val", val_ds), ("test", test_ds)]
    #dss = [("test", test_ds)]

    # 2. Define model
    path = "./paper/poly/"

    models = [("my_inv_fc", GroupInvariance(Z5, 64)), ("conv_my_inv", GroupInvarianceConv(Z5, 116)),
              ("avg_fc", SimpleNet()), ("conv_avg_imp", Conv1d()), ("maron_sw", Maron())]

    results = []
    for base_name, model in models:
        for ds_name, ds in dss:
            mae = []
            mape = []
            times = []
            for i in range(1, 11):
                print("MODEL: ", base_name, ds_name, i)
                best_path = sorted(glob(path + base_name + "_" + str(i) + "/checkpoints/best*.index"),
                                   key=lambda x: (len(x), x))[-1].replace(".index", "")
                model.load_weights(best_path).expect_partial()

                acc = []
                per = []
                for i in range(len(ds)):
                    # 5. Run everything
                    start = time()
                    pred = model(ds[i])
                    if len(pred) == 2:
                        pred, L = pred
                    times.append(time() - start)

                    y = poly_Z5(ds[i])[:, tf.newaxis]
                    model_loss = tf.keras.losses.mean_absolute_error(y, pred)
                    f = model_loss / y
                    acc = acc + list(model_loss.numpy())
                    per = per + list(f.numpy())

                print(np.mean(acc))
                mae.append(np.mean(acc))
                mape.append(np.mean(per))
            results.append((base_name, ds_name, np.mean(mae), np.std(mae), np.mean(times[1:]), np.std(times[1:])))
            print(base_name, ds_name, np.mean(mae), np.std(mae), np.mean(times[1:]), np.std(times[1:]))

    with open("./paper/poly.csv", 'w') as fh:
        for r in results:
            fh.write("%s\t%s\t%.5f\t%.5f\t%.6f\t%.6f\n" % r)


if __name__ == '__main__':
    main()
