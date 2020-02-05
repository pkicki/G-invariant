from math import pi
from random import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

from utils.distances import if_inside, if_collide


def invariance(path, id=0):
    xp = random()
    yp = random()
    d = 0.2
    xc = (1. - d * 2) * random() + d
    yc = (1. - d * 2) * random() + d
    # base = np.array([0., pi/2, pi, 3*pi/2])
    n = 4
    b = 2 / n * pi
    base = np.arange(n) * b
    base = np.roll(base, randint(0, n - 1))
    th = b * np.random.random(n) + base
    r = 0.3 + 0.8 * np.random.random(n)
    xq = r * np.cos(th) + xc
    yq = r * np.sin(th) + yc

    plt.plot(np.concatenate([xq, xq[:1]], -1), np.concatenate([yq, yq[:1]], -1))
    plt.plot(xp, yp, 'r*')

    inside = if_inside(np.stack([xq, yq], -1), np.stack([xp, yp], -1))

    scenario = np.concatenate([xq, yq, [xp], [yp], [int(inside.numpy())]], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # plt.savefig(fname + ".png")
    # plt.clf()

    return int(inside.numpy())


def two_quads(path, id=0):
    d = 0.1
    xc = (1. - d * 2) * random() + d
    yc = (1. - d * 2) * random() + d
    n = 4
    b = 2 / n * pi
    base = np.arange(n) * b
    base = np.roll(base, randint(0, n - 1))
    th = b * np.random.random(n) + base
    r = 0.1 + 0.4 * np.random.random(n)
    x1 = r * np.cos(th) + xc
    y1 = r * np.sin(th) + yc

    d = 0.1
    xc = (1. - d * 2) * random() + d
    yc = (1. - d * 2) * random() + d
    n = 4
    b = 2 / n * pi
    base = np.arange(n) * b
    base = np.roll(base, randint(0, n - 1))
    th = b * np.random.random(n) + base
    r = 0.1 + 0.4 * np.random.random(n)
    x2 = r * np.cos(th) + xc
    y2 = r * np.sin(th) + yc

    plt.plot(np.concatenate([x1, x1[:1]], -1), np.concatenate([y1, y1[:1]], -1))
    plt.plot(np.concatenate([x2, x2[:1]], -1), np.concatenate([y2, y2[:1]], -1))

    dn = 50
    dx1 = np.linspace(x1[0], x1[1], dn)
    dy1 = np.linspace(y1[0], y1[1], dn)
    dx2 = np.linspace(x1[1], x1[2], dn)
    dy2 = np.linspace(y1[1], y1[2], dn)
    dx3 = np.linspace(x1[2], x1[3], dn)
    dy3 = np.linspace(y1[2], y1[3], dn)
    dx4 = np.linspace(x1[3], x1[0], dn)
    dy4 = np.linspace(y1[3], y1[0], dn)
    dx = np.concatenate([dx1, dx2, dx3, dx4], 0)
    dy = np.concatenate([dy1, dy2, dy3, dy4], 0)
    d = np.stack([dx, dy], -1)
    collide = if_collide(np.stack([x2, y2], -1), d)
    plt.show()

    scenario = np.concatenate([x1, y1, x2, y2, [int(collide.numpy())]], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # plt.savefig(fname + ".png")
    # plt.clf()

    print(collide)

    return int(collide.numpy())


def area(path, id=0):
    def area3(xy):
        d = np.diff(np.concatenate([xy, xy[:1]], axis=0), axis=0)
        d = np.sqrt(np.sum(np.square(d), -1))
        p = 1 / 2 * np.sum(d)
        pmd = p - d
        a = np.sqrt(p * np.prod(pmd))
        return a

    def area4(xy):
        A = xy[0]
        B = xy[1]
        C = xy[2]
        D = xy[3]
        AC = A - C
        BD = B - D
        a = 1 / 2 * np.abs(AC[0] * BD[1] - AC[1] * BD[0])
        return a

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
        n = 10
        b = 2 / n * pi
        base = np.arange(n) * b
        base = np.roll(base, randint(0, n - 1))
        th = b * np.random.random(n) + base
        rb = 0.3 + 0.5 * random()
        r = 0.4 * (np.random.random(n) - 0.5) + rb
        xq = r * np.cos(th) + xc
        yq = r * np.sin(th) + yc

        #roll = randint(0, n - 1)
        #xq = np.roll(xq, roll)
        #yq = np.roll(yq, roll)

        #print("NEXT")
        #print(xq)
        #print(yq)
        xq = np.abs(xq)
        yq = np.abs(yq)
        #print(xq)
        #print(yq)
        plt.plot(np.concatenate([xq, xq[:1]], -1), np.concatenate([yq, yq[:1]], -1))
        #plt.show()

        xy = np.stack([xq, yq], axis=-1)
        # a = area3(xy)
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

    #print("COORDS")
    #print(xq)
    #print(yq)
    #plt.show()
    scenario = np.concatenate([xq, yq, [a]], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # plt.savefig(fname + ".png")
    # plt.clf()
    cv2.imwrite(fname + ".png", m)


def inpainting(path, id=0):
    d = 0.4
    xc = (1. - d * 2) * random() + d
    yc = (1. - d * 2) * random() + d
    # base = np.array([0., pi/2, pi, 3*pi/2])
    n = 4
    # b = 2 / n * pi
    # base = np.arange(n) * b
    # base = np.roll(base, randint(0, n - 1))
    # th = b * np.random.random(n) + base
    th = 2 * np.pi * np.random.random(n)
    r = 0.1 + 0.4 * np.random.random(n)
    xq = r * np.cos(th) + xc
    yq = r * np.sin(th) + yc

    # image preparing
    W = 64
    H = 64
    W_range = 1.
    H_range = 1.

    xq_px = np.around(W * xq / W_range)
    yq_px = np.around(H * (1. - yq / H_range))

    poly = np.stack([xq_px, yq_px], -1).astype(np.int32)

    m = np.zeros((W, H), dtype=np.uint8)
    # cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    cv2.arrowedLine(m, tuple(poly[0]), tuple(poly[1]), 255, 1)
    cv2.arrowedLine(m, tuple(poly[1]), tuple(poly[2]), 255, 1)
    cv2.arrowedLine(m, tuple(poly[2]), tuple(poly[3]), 255, 1)
    cv2.arrowedLine(m, tuple(poly[3]), tuple(poly[0]), 255, 1)

    scenario = np.concatenate([xq, yq], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # saves
    cv2.imwrite(fname + ".png", m)
    # plt.savefig(fname + ".png")
    # plt.clf()


def convex(path, id=0):
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

    #convex = False
    #while not convex:
    d = 0.2
    xc = (1. - d * 2) * random() + d
    yc = (1. - d * 2) * random() + d
    # base = np.array([0., pi/2, pi, 3*pi/2])
    # n = 3
    n = 4
    b = 2 / n * pi
    base = np.arange(n) * b
    base = np.roll(base, randint(0, n - 1))
    th = b * np.random.random(n) + base
    rb = 0.3 + 0.5 * random()
    r = 1.4 * (np.random.random(n) - 0.5) + rb
    xq = r * np.cos(th) + xc
    yq = r * np.sin(th) + yc

    xq = np.abs(xq)
    yq = np.abs(yq)
    # plt.plot(np.concatenate([xq, xq[:1]], -1), np.concatenate([yq, yq[:1]], -1))

    xy = np.stack([xq, yq], axis=-1)
    # a = area3(xy)

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

    scenario = np.concatenate([xq, yq, [float(convex)]], -1)

    fname = path + str(id).zfill(6)
    np.savetxt(fname + ".scn", scenario, fmt='%.4f', delimiter='\t')
    # plt.savefig(fname + ".png")
    # plt.clf()
    cv2.imwrite(fname + ".png", m)

    return float(convex)


# path = "../../data_inv/train/quads/"
# path = "../../data_inv/val/quads/"
# path = "../../data_inv/train/quads_permute3/"
# path = "../../data_inv/val/quads_permute3/"
# path = "../../data_inv/train/5/"
# path = "../../data_inv/val/5/"
# path = "../../data_inv/train/quads_area/"
# path = "../../data_inv/val/quads_area/"
# path = "../../data_inv/train/two_quads2/"
# path = "../../data_inv/val/two_quads2/"
# path = "../../data_inv/train/inpainting3/"
# path = "../../data_inv/val/inpainting3/"
# path = "../../data_inv/train/area4d/"
# path = "../../data_inv/test/area4d/"
# path = "../../data_inv/val/area4c/"
# path = "../../data_inv/train/area4f/"
# path = "../../data_inv/val/area4e/"
#path = "../../data_inv/train/area5paper/"
#path = "../../data_inv/train/area7paper_shift/"
path = "../../data_inv/test/area10paper_shift/"
#path = "../../data_inv/"
#path = "../../data_inv/test/two_quads_paper/"
#path = "../../data_inv/train/convexity/"
#path = "../../data_inv/test/convexity/"
# for i in range(256):
inside_n = 0
n = 1024
# n = 8192
#n = 256
#n = 10
for i in range(n):
    # for i in range(2048):
    # inside = invariance(path, i)
    #inside = two_quads(path, i)
    # inpainting(path, i)
    area(path, i)
    #is_convex = convex(path, i)
    #inside_n += is_convex
# inside_n += inside
#print(inside_n / float(n))
