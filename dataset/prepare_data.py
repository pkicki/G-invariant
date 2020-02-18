from math import pi
from random import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()


def area(path, roll, id=0):
    def area(xy):
        def cross_product3(a, b):
            return a[:, :, 0] * b[:, :, 1] - b[:, :, 0] * a[:, :, 1]

        def if_collide(verts, query_points):
            first_point_coords = verts
            second_point_coords = tf.roll(verts, -1, -2)
            edge_vector = second_point_coords - first_point_coords
            query_points_in_v1 = query_points[:, tf.newaxis] - first_point_coords[tf.newaxis]
            cross = cross_product3(edge_vector[tf.newaxis], query_points_in_v1)
            inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
            return inside

        n = int(1e5)
        a = 2.
        pts = 2 * a * np.random.rand(n, 2) - a
        collide = if_collide(xy, pts)
        s = np.sum((tf.cast(collide, tf.float32)).numpy())
        return s / n * (2 * a) ** 2

    convex = False
    while not convex:
        d = 0.2
        xc = (1. - d * 2) * random() + d
        yc = (1. - d * 2) * random() + d
        # base = np.array([0., pi/2, pi, 3*pi/2])
        # n = 3
        n = 4
        b = 2 / n * pi
        base = np.arange(n) * b
        if roll:
            base = np.roll(base, randint(0, n - 1))
        th = b * np.random.random(n) + base
        rb = 0.3 + 0.5 * random()
        r = 0.4 * (np.random.random(n) - 0.5) + rb
        xq = r * np.cos(th) + xc
        yq = r * np.sin(th) + yc

        xq = np.abs(xq)
        yq = np.abs(yq)
        plt.plot(np.concatenate([xq, xq[:1]], -1), np.concatenate([yq, yq[:1]], -1))

        xy = np.stack([xq, yq], axis=-1)
        a = area(xy)

        W = 64
        H = 64
        W_range = 1.5
        H_range = 1.5

        diffx = np.diff(np.concatenate([xq, xq[:1]], axis=0))
        diffy = np.diff(np.concatenate([yq, yq[:1]], axis=0))

        diffx_1 = np.roll(diffx, 1, 0)
        diffy_1 = np.roll(diffy, 1, 0)

        cross = diffx * diffy_1 - diffx_1 * diffy
        convex = np.logical_or(np.all(cross > 0), np.all(cross < 0))

    xq_px = np.around(W * xq / W_range)
    yq_px = np.around(H * ((H_range - yq) / H_range))

    poly = np.stack([xq_px, yq_px], -1).astype(np.int32)

    m = np.zeros((W, H), dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)

    scenario = np.concatenate([xq, yq, [a]], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # plt.savefig(fname + ".png")
    # plt.clf()
    cv2.imwrite(fname + ".png", m)


path = "../../data_inv/train/area4paper_shift/"
for i in range(256):
    area(path, False, i)
path = "../../data_inv/val/area4paper_shift/"
for i in range(256):
    area(path, True, i)
path = "../../data_inv/test/area4paper_shift/"
for i in range(1024):
    area(path, True, i)
