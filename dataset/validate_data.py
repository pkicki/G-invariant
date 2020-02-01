import os
import tensorflow as tf
from math import pi
from random import *
import numpy as np
import cv2

from dataset.scenarios import get_map
from models.invariance_models import invalidate
from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist


def invalidate(x, y, fi, free_space):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    crucial_points = tf.stack(crucial_points, -2)

    penetration = dist(free_space, crucial_points)

    in_obstacle = tf.reduce_sum(penetration, -1)
    violation_level = tf.reduce_sum(in_obstacle, -1)

    # violation_level = integral(env.free_space, crucial_points)
    return violation_level

#path = "../../data/train/tunel/"
path = "../../data/val/tunel/"
#path = "../../data/train/parkowanie_prostopadle/"
#path = "../../data/val/parkowanie_prostopadle/"
def read_scn(scn_path):
    scn_path = os.path.join(path, scn_path)
    data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
    p0 = tf.unstack(data[0][:4], axis=0)
    pk = tf.unstack(data[-1][:3], axis=0)
    return p0, pk


def read_map(map_path):
    map_path = os.path.join(path, map_path)
    return get_map(map_path)


scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]
maps = [read_map(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]

x0 = np.array([s[0][0] for s in scenarios])[:, np.newaxis]
y0 = np.array([s[0][1] for s in scenarios])[:, np.newaxis]
fi0 = np.array([s[0][2] for s in scenarios])[:, np.newaxis]

x1 = np.array([s[1][0] for s in scenarios])[:, np.newaxis]
y1 = np.array([s[1][1] for s in scenarios])[:, np.newaxis]
fi1 = np.array([s[1][2] for s in scenarios])[:, np.newaxis]

free_space = tf.stack(maps, 0)

a0 = list(invalidate(x0, y0, fi0, free_space).numpy())
a1 = list(invalidate(x1, y1, fi1, free_space).numpy())

for i in range(len(a0)):
    if a0[i] > 0 or a1[i] > 0:
        print(i, " ", a0[i], " ", a1[i])
