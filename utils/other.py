import tensorflow as tf


def partitionfunc(n, k, l=1):
    '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
    if k < 1:
        raise StopIteration
    if k == 1:
        if n >= l:
            yield (n,)
        raise StopIteration
    for i in range(l, n + 1):
        for result in partitionfunc(n - i, k - 1, i):
            yield (i,) + result


def groupAvereaging(inputs, operation):
    x = inputs
    x1 = x
    x2 = tf.roll(x, 1, 1)
    x3 = tf.roll(x, 2, 1)
    x4 = tf.roll(x, 3, 1)

    x1 = operation(x1)
    x2 = operation(x2)
    x3 = operation(x3)
    x4 = operation(x4)

    x = tf.reduce_mean(tf.stack([x1, x2, x3, x4], -1), -1)
    return x


def apply_layers(x, layers):
    for l in layers:
        x = l(x)
    return x
