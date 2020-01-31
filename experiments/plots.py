from glob import glob
import numpy as np
import matplotlib.pyplot as plt


def tsplot(data, label):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)
    line, = plt.plot(m)
    line.set_label(label)
    plt.fill_between(np.arange(len(m)), m + s, m - s, alpha=0.5)


start = 0
end = 70
#names = ["avg_fc", "my_inv_fc"]
#names = ["conv_avg", "fc_avg", "my_inv_fc_none_tanh", "my_inv_conv_none_tanh", "maron_smaller_mulnet", "conv_img"]
names = ["maron_sw", "conv_my_inv"]
plot_names = ["Maron", "Conv1D G-inv (ours)"]
path = "./paper/poly/"
#path = "./working_dir/poly/"
for k, base_name in enumerate(names):
    seq = []
    for i in range(1, 11):
        training_path = glob(path + base_name + "_" + str(i) + "/*.csv")[0]
        data = np.genfromtxt(training_path, delimiter=",", dtype=None)
        data = np.array([x[2] for x in data])
        data = np.reshape(data, (-1, 2))
        seq.append(data)

    data = np.stack(seq, axis=0)
    tsplot(data[:, start:end, 0], plot_names[k] + " TRAIN")
    tsplot(data[:, start:end, 1], plot_names[k] + " VAL")
plt.xlabel("Number of epochs")
plt.ylabel("MAE")
plt.legend()
plt.show()
