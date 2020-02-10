#!/usr/bin/python
import tensorflow as tf

tf.enable_eager_execution()

def dist2vert(v, q):
    v = tf.expand_dims(v, 1)
    q = tf.expand_dims(q, 2)
    d = euclid(v, q)
    d = tf.reduce_min(d, 2)
    return d


def euclid(a, b=None):
    if b is None:
        return tf.sqrt(tf.reduce_sum(a ** 2, -1))
    return tf.sqrt(tf.reduce_sum((a - b) ** 2, -1))


def cross_product(a, b):
    return a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]

def cross_product3(a, b):
    return a[:, :, 0] * b[:, :, 1] - b[:, :, 0] * a[:, :, 1]


def point2edge(verts, query_points):
    first_point_coords = verts
    second_point_coords = tf.roll(verts, -1, -2)
    edge_vector = second_point_coords - first_point_coords
    edge_vector = edge_vector[:, tf.newaxis, tf.newaxis]
    query_points_in_v1 = query_points[:, :, :, tf.newaxis, tf.newaxis] - first_point_coords[:, tf.newaxis, tf.newaxis]
    p = tf.reduce_sum(edge_vector * query_points_in_v1, -1)
    cross = cross_product(edge_vector, query_points_in_v1)
    inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
    inside = tf.reduce_any(inside, -1)
    t = tf.reduce_sum(edge_vector * edge_vector, -1)
    w = p / (t + 1e-8)
    w = tf.where(w <= 0, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    w = tf.where(w >= 1, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    p = edge_vector * tf.expand_dims(w, -1) \
        + first_point_coords[:, tf.newaxis, tf.newaxis]  # calcualte point on the edge
    return p - query_points[:, :, :, tf.newaxis, tf.newaxis], inside


def point2vert(verts, query_points):
    return verts[:, tf.newaxis, tf.newaxis] - query_points[:, :, :, tf.newaxis, tf.newaxis]


def dist(verts, query_points):
    """

    :param verts: (N, V, 4, 2)
    :param query_points: (N, S, P, 2)
    :return:
    """
    p2e, inside = point2edge(verts, query_points)
    p2v = point2vert(verts, query_points)
    p2e = tf.reduce_sum(tf.abs(p2e), axis=-1)
    p2v = euclid(p2v)
    dists = tf.concat([p2e, p2v], -1)
    dists = tf.reduce_min(dists, (-2, -1))
    dists = tf.where(inside, tf.zeros_like(dists), dists)
    return dists


import matplotlib.pyplot as plt


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


def integral(free_space, path):
    segment_centers = (path[:, 1:] + path[:, :-1]) / 2.
    segments_direction = path[:, 1:] - path[:, :-1]
    segments_lengths = tf.linalg.norm(segments_direction, axis=-1, keep_dims=True)
    segments_perpendicular_direction = tf.reverse(segments_direction, [-1]) / (segments_lengths + 1e-8)
    perp_line = segments_perpendicular_direction[:, tf.newaxis] * tf.linspace(0.0, 1.0, 2)[tf.newaxis, :, tf.newaxis,
                                                                  tf.newaxis, tf.newaxis]
    p = segment_centers[:, tf.newaxis] + perp_line
    x1 = p[:, 0, :, :, 0]
    x2 = p[:, 1, :, :, 0]
    y1 = p[:, 0, :, :, 1]
    y2 = p[:, 1, :, :, 1]
    Ap = y2 - y1
    Bp = x1 - x2
    Cp = x2 * y1 - x1 * y2

    first_point_coords = free_space
    second_point_coords = tf.roll(free_space, -1, -2)
    free_space_segments = tf.stack([first_point_coords, second_point_coords], -2)

    Ap = Ap[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    Bp = Bp[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    Cp = Cp[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    free_space_segments = free_space_segments[:, tf.newaxis, tf.newaxis]
    xs = free_space_segments[:, :, :, :, :, :, 0]
    ys = free_space_segments[:, :, :, :, :, :, 1]
    if_on_the_other_sides = Ap * xs + Bp * ys + Cp
    if_on_the_other_sides = tf.less(tf.reduce_prod(if_on_the_other_sides, -1), 0)

    x1 = xs[:, :, :, :, :, 0]
    y1 = ys[:, :, :, :, :, 0]
    x2 = xs[:, :, :, :, :, 1]
    y2 = ys[:, :, :, :, :, 1]
    As = y2 - y1
    Bs = x1 - x2
    Cs = x2 * y1 - x1 * y2

    segment_centers = segment_centers[:, :, :, tf.newaxis, tf.newaxis]
    xsc = segment_centers[:, :, :, :, :, 0]
    ysc = segment_centers[:, :, :, :, :, 1]
    inside = As * xsc + Bs * ysc + Cs
    inside = inside < 0
    inside = tf.reduce_all(inside, -1)

    D = tf.squeeze(Ap, -1) * Bs - As * tf.squeeze(Bp, -1)
    D = tf.where(if_on_the_other_sides, D, 1e-8 * tf.ones_like(D))
    xc = (tf.squeeze(Bp, -1) - Bs) / D
    yc = -(tf.squeeze(Ap, -1) - As) / D

    xyc = tf.stack([xc, yc], -1)
    dist = tf.linalg.norm(xyc - segment_centers, axis=-1)
    dist = dist * tf.cast(if_on_the_other_sides, tf.float32)
    dist = tf.where(if_on_the_other_sides, dist, 1e10 * tf.ones_like(dist))

    dist = tf.reduce_min(dist, -1)
    dist = tf.where(inside, tf.zeros_like(dist), dist)
    dist = tf.reduce_min(dist, -1)

    penetration = dist * segments_lengths[:, :, :, 0]
    penetration = tf.reduce_mean(penetration, -1)

    return tf.reduce_sum(penetration, -1)
