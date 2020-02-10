import tensorflow as tf
from utils.constants import Car


def calculate_car_crucial_points(x, y, fi):
    pose = tf.stack([x, y], -1)
    cfi = tf.cos(fi)
    sfi = tf.sin(fi)
    cs = tf.stack([cfi, sfi], -1)
    msc = tf.stack([-sfi, cfi], -1)
    front_center = pose + Car.rear_axle_to_front * cs
    back_center = pose - Car.rear_axle_to_back * cs
    front_left = front_center + msc * Car.W / 2
    front_right = front_center - msc * Car.W / 2
    back_left = back_center + msc * Car.W / 2
    back_right = back_center - msc * Car.W / 2
    return [pose, front_left, front_right, back_left, back_right]
