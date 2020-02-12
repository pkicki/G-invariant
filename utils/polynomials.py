import tensorflow as tf


def poly_Z5(x):
    def inv1(a, b):
        return a * b ** 2

    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = inv1(a, b) + inv1(b, c) + inv1(c, d) + inv1(d, e) + inv1(e, a)
    return q1


def poly_D8(x):
    def inv1(a, b):
        return a * b ** 2

    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = inv1(a, b) + inv1(b, c) + inv1(c, d) + inv1(d, a) + \
         inv1(b, a) + inv1(c, b) + inv1(d, c) + inv1(a, d)
    return q1 + e


def poly_A4(x):
    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = a * b + c * d
    q2 = a * c + b * d
    q3 = a * d + b * c
    q4 = a * b * c + a * b * d + a * c * d + b * c * d

    return q1 + q2 + q3 + q4 + e


def poly_S4(x):
    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = a * b * c * d
    return q1 + e


def poly_S3xS2(x):
    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = a*b*c + d + e
    return q1

def poly_S3(x):
    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = a * b * c + 2 * d + e
    return q1

def poly_Z3(x):
    def inv1(a, b):
        return a * b ** 2

    a, b, c, d, e = tf.unstack(x, axis=1)
    q1 = inv1(a, b) + inv1(b, c) + inv1(c, a) + 2 * d + e
    return q1
