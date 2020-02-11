import os
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def quadrangle_area_dataset(path):
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
