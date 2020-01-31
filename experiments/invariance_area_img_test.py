import inspect
import os
import sys
from glob import glob

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.area4 import GroupInvariance, SimpleNet, Conv1d, SegmentNet, GroupInvarianceConv, \
    ConvImg

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
# tf.set_random_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data[0], data[1], data[2])
            pbar.update(batch_size)


def main(args):
    # 1. Get datasets
    train_ds, train_size = scenarios.area_img_dataset(args.scenario_path.replace("train", "test"))
    #train_ds, train_size = scenarios.area_img_dataset(args.scenario_path.replace("train", "val"))
    #train_ds, train_size = scenarios.area_img_dataset(args.scenario_path)

    # 2. Define model
    n = 32
    model = ConvImg(n)

    base_name = "conv_img"
    path = "./working_dir/area/"
    mae = []

    for i in range(1, 10):
        best_path = sorted(glob(path + base_name + "_" + str(i) + "/checkpoints/best*.index"),
                           key=lambda x: (len(x), x))[-1].replace(".index", "")
        model.load_weights(best_path).expect_partial()

        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

        acc = []
        for i, quad, area, img in _ds('Train', dataset_epoch, train_size, 0, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                pred = model(img[:, :, :, tf.newaxis], training=True)

                model_loss = tf.keras.losses.mean_absolute_error(area[:, tf.newaxis], pred)

            acc = acc + list(model_loss.numpy())

        print(np.mean(acc))
        mae.append(np.mean(acc))

    print(np.mean(mae))
    print(np.std(mae))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--train-beta', type=float, default=0.99)
    args, _ = parser.parse_known_args()
    main(args)
