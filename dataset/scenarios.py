import os

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from utils.utils import Pose2D
from PIL import Image

tf.enable_eager_execution()


def two_quads_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:4]
        y = data[4:8]
        xy1 = tf.stack([x, y], -1)
        x = data[8:12]
        y = data[12:16]
        xy2 = tf.stack([x, y], -1)
        if_collide = data[-1:]
        return xy1, xy2, if_collide

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1], scenarios[i][2]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def invariance_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:4]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        xyp = data[8:-1]
        if_inside = data[-1:]
        return xy, xyp, if_inside

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1], scenarios[i][2]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def decode_data(data):
    p0 = data[:, :1]
    pk = data[:, 1:]
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    xk, yk, thk = tf.unstack(pk, axis=-1)
    return x0, y0, th0, xk, yk, thk


def convexity_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # x = data[:3]
        x = data[:4]
        # y = data[3:6]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        # area = data[6]
        convex = data[8]
        return xy, convex

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def area_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # x = data[:3]
        # x = data[:4]
        # x = data[:5]
        # x = data[:6]
        x = data[:7]
        # y = data[3:6]
        # y = data[4:8]
        # y = data[5:10]
        # y = data[6:12]
        y = data[7:14]
        xy = tf.stack([x, y], -1)
        # area = data[6]
        # area = data[8]
        # area = data[10]
        # area = data[12]
        area = data[14]
        return xy, area

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def area6_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # x = data[:3]
        # x = data[:4]
        # x = data[:5]
        x = data[:6]
        # x = data[:7]
        # y = data[3:6]
        # y = data[4:8]
        # y = data[5:10]
        y = data[6:12]
        # y = data[7:14]
        xy = tf.stack([x, y], -1)
        # area = data[6]
        # area = data[8]
        # area = data[10]
        area = data[12]
        # area = data[14]
        return xy, area

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def area5_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:5]
        y = data[5:10]
        xy = tf.stack([x, y], -1)
        area = data[10]
        return xy, area

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def area4_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:4]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        area = data[8]
        return xy, area

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def area_img_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # x = data[:3]
        x = data[:4]
        # y = data[3:6]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        # area = data[6]
        area = data[8]
        png_path = scn_path.replace("scn", "png")
        img = mpimg.imread(png_path)
        return xy, area, img

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1], scenarios[i][2]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)


def inpainting_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        x = data[:4]
        y = data[4:8]
        xy = tf.stack([x, y], -1)
        png_path = scn_path.replace("scn", "png")
        img = mpimg.imread(png_path)
        return xy, img

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]

    def gen():
        for i in range(len(scenarios)):
            yield scenarios[i][0], scenarios[i][1]

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios)
