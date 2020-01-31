import inspect
import os
import sys
from glob import glob
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from experiments.invariance_poly import poly_simple
from models.poly import GroupInvariance, SimpleNet, MessagePassing, Maron, Conv1d, GroupInvarianceConv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
# tf.set_random_seed(444)
np.random.seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def main(args):
    # 1. Get datasets
    ts = int(1e0)
    vs = int(3e1)
    s = int(3e2)

    d = 5
    train_ds = np.random.rand(ts, args.batch_size, d)
    val_ds = np.random.rand(vs, args.batch_size, d)
    test_ds = np.random.rand(s, args.batch_size, d)

    # 2. Define model
    n = 32
    #model = GroupInvariance(n)
    model = GroupInvarianceConv(n)
    #model = SimpleNet(n)
    #model = MessagePassing(n)
    #model = Maron(n)
    #model = Conv1d(n)

    #ds = train_ds
    ds = val_ds
    #ds = test_ds

    #base_name = "my_inv_fc"
    #base_name = "avg_fc"
    #base_name = "conv_avg_imp"
    base_name = "conv_my_inv"
    #base_name = "maron"
    #base_name = "maron_sw"
    #base_name = "message_passing"
    path = "./working_dir/poly/"
    mae = []
    times = []
    for i in range(1, 11):
        best_path = sorted(glob(path + base_name + "_" + str(i) + "/checkpoints/best*.index"),
                           key=lambda x: (len(x), x))[-1].replace(".index", "")
        model.load_weights(best_path).expect_partial()

        acc = []
        for i in range(len(ds)):
            # 5. Run everything
            start = time()
            #pred, L = model(ds[i], training=True)
            pred = model(ds[i], training=True)
            times.append(time() - start)
            y = poly_simple(ds[i])[:, tf.newaxis]
            model_loss = tf.keras.losses.mean_absolute_error(y, pred)
            acc = acc + list(model_loss.numpy())

        print(np.mean(acc))
        mae.append(np.mean(acc))

    print("RESULTS:")
    print(np.mean(mae))
    print(np.std(mae))
    print(times[1:])
    print(len(times[1:]))
    print(np.mean(times[1:]))
    print(np.std(times[1:]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--train-beta', type=float, default=0.99)
    args, _ = parser.parse_known_args()
    main(args)
