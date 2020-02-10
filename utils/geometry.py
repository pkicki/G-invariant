#!/usr/bin/python
import tensorflow as tf

tf.enable_eager_execution()


def cross_product(a, b):
    return a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]


def cross_product3(a, b):
    return a[:, :, 0] * b[:, :, 1] - b[:, :, 0] * a[:, :, 1]


def if_inside(verts, query_points):
    first_point_coords = verts
    second_point_coords = tf.roll(verts, -1, -2)
    edge_vector = second_point_coords - first_point_coords
    query_points_in_v1 = query_points[tf.newaxis] - first_point_coords
    cross = cross_product(edge_vector, query_points_in_v1)
    inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
    return inside


def if_collide(verts, query_points):
    first_point_coords = verts
    second_point_coords = tf.roll(verts, -1, -2)
    edge_vector = second_point_coords - first_point_coords
    query_points_in_v1 = query_points[:, tf.newaxis] - first_point_coords[tf.newaxis]
    cross = cross_product3(edge_vector[tf.newaxis], query_points_in_v1)
    inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
    inside = tf.reduce_any(inside)
    return inside


