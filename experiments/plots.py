from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def tsplot(data, label):
    m = np.mean(data, axis=0)
    s = np.std(data, axis=0)

    m = savgol_filter(m, 41, 5)

    line, = plt.plot(m)
    line.set_label(label)
    #plt.fill_between(np.arange(len(m)), m + s, m - s, alpha=0.5)


start = 0
end = 1250
#end = 50
#names = ["avg_fc", "my_inv_fc"]
#names = ["conv_avg", "fc_avg", "my_inv_fc_none_tanh", "my_inv_conv_none_tanh", "maron_smaller_mulnet", "conv_img"]
#names = ["maron_sw", "conv_my_inv"]
names = ["Z3", "S3", "S3xS2"]
plot_names = [r"$\mathbb{Z}_3$", r"$S_3$", r"$S_3 \times S_2$"]
#path = "./paper/poly/"
#path = "./working_dir/poly_GH_tr160/"
path = "./paper/poly_GH/"
for k, base_name in enumerate(names):
    seq = []
    for i in range(1, 11):
        training_path = glob(path + base_name + "_" + str(i) + "/*.csv")[0]
        data = np.genfromtxt(training_path, delimiter=",", dtype=None)
        data = np.array([x[2] for x in data])
        #data = np.reshape(data, (-1, 2))
        data = np.reshape(data, (-1, 4))
        seq.append(data)

    data = np.stack(seq, axis=0)
    print(data.shape)
    #tsplot(data[:, start:end, 1], plot_names[k] + " TRAIN")
    #tsplot(data[:, start:end, 3], plot_names[k] + " VAL")
    tsplot(data[:, start:end, 1], plot_names[k] + " TRAIN")
    tsplot(data[:, start:end, 3], plot_names[k] + " VAL")
plt.xlabel("Number of epochs")
#plt.ylabel("MAE")
plt.ylabel("MAPE")
plt.legend()
#plt.ylim(0.0, 0.1)
#plt.yscale("log")
plt.ylim(0.0, 0.1)
plt.show()
