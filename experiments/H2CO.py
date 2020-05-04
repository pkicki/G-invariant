import inspect
import os
import sys
import numpy as np


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from models.H2CO import *
from utils.permutation_groups import Z4, S3_in_S4, H3O_perm, H2CO_perm

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
# tf.set_random_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data[0], data[1])
            pbar.update(batch_size)
            
K = 219474.6


def main(args):
    # 1. Get datasets
    train_ds, train_size, val_ds, val_size, test_ds, test_size, m, std = scenarios.H2CO_pos_dataset(args.scenario_path)

    val_bs = args.batch_size
    val_ds = val_ds \
        .batch(val_bs) \
        .prefetch(val_bs)

    # 2. Define model


    perm = H2CO_perm
    ##model = GroupInvariance(perm, 256)
    model = GroupInvariance(perm)
    #model = FCGroupAvg()
    #model = FC()
    #model = Pooling()
    #model = Maron()
    #model(tf.zeros((args.batch_size, 6, 3), dtype=tf.float32))
    #model.load_weights("./working_dir/chemistry/H2CO_my/checkpoints/last_n-349").assert_consumed()

    # 3. Optimization
    optimizer = tf.train.AdamOptimizer(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)
    #experiment_handler.restore("./working_dir/chemistry/H2CO_my/checkpoints/last_n-349").assert_consumed()

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 1e10
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        rmse = tf.keras.metrics.RootMeanSquaredError()
        for i, xyz, energy, in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                pred = model(xyz)
                if len(pred) == 2:
                    L = pred[1]
                    pred = pred[0]
                else:
                    L = 0

                ## uncomment to check model size
                #nw = 0
                #for layer in model.layers:
                #   for l in layer.get_weights():
                #       a = 1
                #       for s in l.shape:
                #           a *= s
                #       nw += a
                #print(nw)

                a = (energy * std + m) * K
                b = (pred * std + m) * K

                #model_loss = tf.keras.losses.mean_absolute_error(a, b)
                model_loss = tf.keras.losses.mean_squared_error(energy, pred)
                #reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = model_loss + L
                rmse.update_state(a, b)

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            acc = acc + list(model_loss.numpy())

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)
                tfc.summary.scalar('metrics/rmse', rmse.result(), step=train_step)

            # 5.1.5 Update meta variables
            train_step += 1

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', np.mean(acc), step=epoch)
            tfc.summary.scalar('epoch/rmse', rmse.result(), step=epoch)

        # 5.2. Validation Loop
        experiment_handler.log_validation()
        acc = []
        rmse = tf.keras.metrics.RootMeanSquaredError()
        for i, xyz, energy, in _ds('Val', val_ds, val_size, epoch, args.batch_size):
            pred = model(xyz)
            if len(pred) == 2:
                L = pred[1]
                pred = pred[0]
            else:
                L = 0

            a = (energy * std + m) * K
            b = (pred * std + m) * K

            model_loss = tf.keras.losses.mean_squared_error(energy, pred)
            rmse.update_state(a, b)

            acc = acc + list(model_loss.numpy())

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)
                tfc.summary.scalar('metrics/rmse', rmse.result(), step=val_step)

            # 5.1.5 Update meta variables
            val_step += 1

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', np.mean(acc), step=epoch)
            tfc.summary.scalar('epoch/rmse', rmse.result(), step=epoch)

        model.save_weights(args.working_path + "/" + args.out_name + "/checkpoints/last_n-" + str(epoch))

        experiment_handler.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--model', type=str)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--n', type=int, default=2)
    args, _ = parser.parse_known_args()
    main(args)
