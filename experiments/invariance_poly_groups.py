import inspect
import os
import sys
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from utils.polynomials import *
from utils.permutation_groups import *
from models.poly import GroupInvariance
# add parent (root) to pythonpath
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
# tf.set_random_seed(444)
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


def main(args):
    # 1. Get datasets
    #ts = int(1e0)
    ts = int(args.ts)
    vs = int(3e1)

    d = 5
    train_ds = np.random.rand(ts, args.batch_size, d)
    val_ds = np.random.rand(vs, args.batch_size, d)

    perm = None
    if args.group == "Z5":
        perm = Z5
    elif args.group == "D8":
        perm = D8
    elif args.group == "A4":
        perm = A4
    elif args.group == "S4":
        perm = S4
    elif args.group == "S3":
        perm = S3
    else:
        print("UNDEFINED GROUP")

    poly = None
    if args.poly == "Z5":
        poly = poly_Z5
    elif args.poly == "D8":
        poly = poly_D8
    elif args.poly == "A4":
        poly = poly_A4
    elif args.poly == "S4":
        poly = poly_S4
    elif args.poly == "S3":
        poly = poly_S3
    elif args.poly == "Z3":
        poly = poly_Z3
    elif args.poly == "S3xS2":
        poly = poly_S3xS2
    else:
        print("UNDEFINED GROUP")

    # 2. Define model
    model = GroupInvariance(perm, args.n)

    # 3. Optimization
    optimizer = tf.train.AdamOptimizer(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 1e10
    for epoch in range(args.num_epochs):
        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        per = []
        for i in tqdm(range(ts), "Train"):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                pred = model(train_ds[i])
                L = 0.
                if len(pred) == 2:
                    pred, L = pred

                ## check model size
                if False:
                    nw = 0
                    for layer in model.layers:
                        for l in layer.get_weights():
                            a = 1
                            for s in l.shape:
                                a *= s
                            nw += a
                    print(nw)

                y = poly(train_ds[i])
                model_loss = tf.keras.losses.mean_absolute_error(y[:, tf.newaxis], pred)
                percent = model_loss / y
                reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = model_loss + L

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            acc = acc + list(model_loss.numpy())
            per = per + list(percent.numpy())

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)

            # 5.1.5 Update meta variables
            train_step += 1

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', np.mean(acc), step=epoch)

        with open(args.working_path + "/" + args.out_name + "/" + model.name + ".csv", 'a') as fh:
            fh.write("TRAIN, %d, %.6f\n" % (epoch, np.mean(acc)))
            fh.write("TRAIN, %d, %.6f\n" % (epoch, np.mean(per)))

        #    accuracy.result()

        # 5.2. Validation Loop
        experiment_handler.log_validation()
        acc = []
        per = []
        for i in tqdm(range(vs), "Val"):
            # 5.2.1 Make inference of the model for validation and calculate losses
            pred = model(val_ds[i])
            if len(pred) == 2:
                pred, L = pred

            y = poly(val_ds[i])
            model_loss = tf.keras.losses.mean_absolute_error(y[:, tf.newaxis], pred)
            percent = model_loss / y

            acc = acc + list(model_loss.numpy())
            per = per + list(percent.numpy())

            # 5.2.3 Print logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1

        epoch_accuracy = np.mean(acc)
        # 5.2.5 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', epoch_accuracy, step=epoch)

        with open(args.working_path + "/" + args.out_name + "/" + model.name + ".csv", 'a') as fh:
            fh.write("VAL, %d, %.6f\n" % (epoch, epoch_accuracy))
            fh.write("VAL, %d, %.6f\n" % (epoch, np.mean(per)))

        # 5.3 Save last and best
        if epoch_accuracy < best_accuracy:
            model.save_weights(args.working_path + "/" + args.out_name + "/checkpoints/best-" + str(epoch))
            best_accuracy = epoch_accuracy

        experiment_handler.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--poly', type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--ts', type=int, default=1)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    args, _ = parser.parse_known_args()
    main(args)
