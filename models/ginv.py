import numpy as np
import tensorflow as tf


def sigmaPi(fin, m, n, p):
    fin = tf.transpose(fin, (0, 2, 1, 3))
    fin = fin[:, :, tf.newaxis]
    fin = tf.tile(fin, (1, 1, m, 1, 1))
    y = fin @ p
    y = tf.linalg.diag_part(y)

    # TO METE:
    # here you can switch to other merging procedures in SigmaPi
    #y = tf.reduce_prod(y, axis=3) ** (1 / n) # different idea
    #y = tf.reduce_prod(y, axis=3) # as in the ICML paper
    y = tf.keras.activations.tanh(tf.reduce_sum(y, axis=3)) # works best on MPII Human Poses

    y = tf.reduce_sum(y, axis=2)
    return y


def prepare_permutation_matices(perm, n, m):
    p1 = np.eye(n, dtype=np.float32)
    p = np.tile(p1[np.newaxis], (m, 1, 1))
    for i, x in enumerate(perm):
        p[i, x, :] = p1[np.arange(n)]
    return p
