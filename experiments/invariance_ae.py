import inspect
import os
import sys
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from models.ae import Decoder, GroupInvarianceConv

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
tf.set_random_seed(444)

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
    train_ds, train_size = scenarios.invariance_dataset(args.scenario_path)
    val_ds, val_size = scenarios.invariance_dataset(args.scenario_path.replace("train", "val"))

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    n = 4
    #model = GroupInvariance(n)
    model = GroupInvarianceConv(n)
    #model = SimpleNet(n)
    #model = Conv1d(n)
    #model = SegmentNet(n)
    decoder = Decoder(128)
    #model = SimpleNet()


    # 3. Optimization

    eta = tfc.eager.Variable(args.eta)
    eta_f = tf.train.exponential_decay(
        args.eta,
        tf.train.get_or_create_global_step(),
        int(float(train_size) / args.batch_size),
        args.train_beta)
    eta.assign(eta_f())
    optimizer = tf.train.AdamOptimizer(eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)


    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
        #dataset_epoch = train_ds

        # 5.1. Training Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        accuracy_90 = tfc.eager.metrics.Accuracy('metrics/accuracy_90')
        experiment_handler.log_training()
        acc = []
        for i, quad, point, inside in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                features = model(quad, training=True)

                ## check model size
                #nw = 0
                #for layer in model.layers:
                #    for l in layer.get_weights():
                #        a = 1
                #        for s in l.shape:
                #            a *= s
                #        nw += a
                #print(nw)

                output = decoder(features, training=True)
                output = tf.reshape(output, (-1, 4, 2))
                o1 = tf.roll(output, 0, 1)
                o2 = tf.roll(output, 1, 1)
                o3 = tf.roll(output, 2, 1)
                o4 = tf.roll(output, 3, 1)

                l1 = tf.reduce_sum((quad - o1)**2, (1, 2))
                l2 = tf.reduce_sum((quad - o2)**2, (1, 2))
                l3 = tf.reduce_sum((quad - o3)**2, (1, 2))
                l4 = tf.reduce_sum((quad - o4)**2, (1, 2))

                model_loss = tf.reduce_min(tf.stack([l1, l2, l3, l4], -1), -1)
                reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = model_loss#+ reg_loss

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            acc = acc + list(model_loss.numpy())

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)
                tfc.summary.scalar('metrics/reg_loss', reg_loss, step=train_step)
                tfc.summary.scalar('training/eta', eta, step=train_step)

            # 5.1.5 Update meta variables
            eta.assign(eta_f())
            train_step += 1
            #if train_step % 20 == 0:
            #    _plot(x_path, y_path, th_path, env, train_step)
            #_plot(x_path, y_path, th_path, data, train_step)
        #epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', np.mean(acc), step=epoch)

        #    accuracy.result()

        # 5.2. Validation Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        experiment_handler.log_validation()
        acc = []
        for i, quad, point, inside in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
            # 5.2.1 Make inference of the model for validation and calculate losses
            features = model(quad, training=True)
            output = decoder(features, training=True)
            output = tf.reshape(output, (-1, 4, 2))
            o1 = tf.roll(output, 0, 1)
            o2 = tf.roll(output, 1, 1)
            o3 = tf.roll(output, 2, 1)
            o4 = tf.roll(output, 3, 1)

            l1 = tf.reduce_sum((quad - o1) ** 2, (1, 2))
            l2 = tf.reduce_sum((quad - o2) ** 2, (1, 2))
            l3 = tf.reduce_sum((quad - o3) ** 2, (1, 2))
            l4 = tf.reduce_sum((quad - o4) ** 2, (1, 2))

            model_loss = tf.reduce_min(tf.stack([l1, l2, l3, l4], -1), -1)

            acc = acc + list(model_loss.numpy())

            # 5.2.3 Print logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1

        epoch_accuracy = np.mean(acc)
        # 5.2.5 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/accuracy', epoch_accuracy, step=epoch)

        print(epoch_accuracy)
        #break

        # 5.3 Save last and best
        if epoch_accuracy > best_accuracy:
            experiment_handler.save_best()
            best_accuracy = epoch_accuracy
        experiment_handler.save_last()

        experiment_handler.flush()



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
