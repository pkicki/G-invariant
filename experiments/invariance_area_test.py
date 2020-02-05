import inspect
import os
import sys
from glob import glob
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from models.area4_small import GroupInvariance, SimpleNet, Conv1d, SegmentNet, GroupInvarianceConv, Maron, \
    MessagePassing

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


names = ["avg_conv", "avg_fc", "message_passing", "my_inv_conv", "my_inv_fc", "maron", "segment"]
#names = ["avg_conv", "avg_fc"]
n = 32
models = [Conv1d(n), SimpleNet(n), MessagePassing(n), GroupInvarianceConv(n), GroupInvariance(n), Maron(n), SegmentNet(n)]


def secondary(args):
    results = []
    area_dataset = None
    if args.n == 4:
        area_dataset = scenarios.area4_dataset
    elif args.n == 5:
        area_dataset = scenarios.area5_dataset
    elif args.n == 6:
        area_dataset = scenarios.area6_dataset
    elif args.n == 7:
        area_dataset = scenarios.area7_dataset

    for i, name in enumerate(names):
        print(name)
        for ds_type in ["train", "val", "test"]:
            scenario_path = "../../data_inv/train/area" + str(args.n) + "paper_shift"
            ds, ds_size = area_dataset(scenario_path.replace("train", ds_type))
            model = models[i]
            path = "./working_dir/area" + str(args.n)
            mae = []
            times = []
            for k in range(1, 10):
                best_path = sorted(glob(path + "/" + name + "_" + str(k) + "/checkpoints/best*.index"),
                                   key=lambda x: (len(x), x))[-1].replace(".index", "")
                model.load_weights(best_path).expect_partial()
                dataset_epoch = ds.shuffle(ds_size)
                dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

                acc = []
                for l, quad, area, in _ds('Train', dataset_epoch, ds_size, 0, args.batch_size):
                    # pred, L = model(quad, training=True)
                    start = time()
                    pred = model(quad, training=True)
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
    with open("./paper/area" + str(args.n) + "_times.csv", 'w') as fh:
        for r in results:
            fh.write("%s\t%s\t%.5f\t%.5f\t%.6f\t%.6f\n" % r)
            #fh.write("%s\t%s\t%.5f\t%.5f\n" % r)



def main(args):
    args.batch_size = 10
    # 1. Get datasets
    # train_ds, train_size = scenarios.area_dataset(args.scenario_path.replace("train", "test"))
    # train_ds, train_size = scenarios.area_dataset(args.scenario_path.replace("train", "val"))
    train_ds, train_size = scenarios.area_dataset(args.scenario_path)

    # 2. Define model
    n = 32
    # model = PolyInvariance(n)
    # model = GroupInvariance(n)
    # model = GroupInvarianceConv(n)
    # model = SimpleNet(n)
    # model = Conv1d(n)
    # model = SegmentNet(n)
    model = Maron(n)
    # model = MessagePassing(n)

    # base_name = "my_inv_conv_none_tanh"
    # base_name = "my_inv_fc_none_tanh"
    # base_name = "conv_avg_sw"
    # base_name = "conv_avg_imp"
    # base_name = "fc_avg"
    # base_name = "segment_net"
    # base_name = "maron_smaller_mulnet"
    base_name = "maron_imp"
    # base_name = "message_passing"
    # path = "./working_dir/area/"
    path = "./paper/area_randomshiftinvalandtest/"
    mae = []
    for i in range(1, 10):
        best_path = sorted(glob(path + base_name + "_" + str(i) + "/checkpoints/best*.index"),
                           key=lambda x: (len(x), x))[-1].replace(".index", "")
        # 4. Restore, Log & Save
        # model.load_weights("./working_dir/area/tmp/checkpoints/last_n-15")
        # model.load_weights("./working_dir/area/my_inv_fc/checkpoints/last_n-200").expect_partial()
        # model.load_weights("./working_dir/area/my_inv_conv_none_tanh_2/checkpoints/last_n-200").expect_partial()
        # model.load_weights("./working_dir/area/my_inv_conv_none_tanh_2/checkpoints/best-228").expect_partial()
        # model.load_weights("./working_dir/area/conv_avg_2/checkpoints/last_n-200").expect_partial()
        # model.load_weights("./working_dir/area/conv_avg_2/checkpoints/best-269").expect_partial()
        model.load_weights(best_path).expect_partial()

        # 5. Run everything
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
        # dataset_epoch = train_ds

        # 5.1. Training Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        accuracy_90 = tfc.eager.metrics.Accuracy('metrics/accuracy_90')
        acc = []
        for i, quad, area, in _ds('Train', dataset_epoch, train_size, 0, args.batch_size):
            pred, L = model(quad, training=True)
            # pred = model(quad, training=True)

            model_loss = tf.keras.losses.mean_absolute_error(area[:, tf.newaxis], pred)
            # model_loss = model_loss / area
            acc = acc + list(model_loss.numpy())

        print(np.mean(acc))
        mae.append(np.mean(acc))

    print(np.mean(mae))
    print(np.std(mae))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n', type=int)
    args, _ = parser.parse_known_args()
    # main(args)
    secondary(args)
