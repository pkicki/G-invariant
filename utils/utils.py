import tensorflow as tf


class Pose2D:
    def __init__(self, x, y, fi):
        self.x = x
        self.y = y
        self.fi = fi

    def get_coords(self):
        return self.x, self.y, self.fi


def Rot(fi):
    c = tf.cos(fi)
    s = tf.sin(fi)
    L = tf.stack([c, s], -1)
    R = tf.stack([-s, c], -1)
    return tf.stack([L, R], -1)


def angleFromRot(R):
    return tf.atan2(R[:, 1, 0], R[:, 0, 0])


def _calculate_length(x, y):
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    lengths = tf.sqrt(dx ** 2 + dy ** 2)
    length = tf.reduce_sum(lengths, -1)
    return length, lengths


class Environment:

    def __init__(self, free_space, max_curvature):
        self.free_space = free_space
        self.max_curvature = max_curvature