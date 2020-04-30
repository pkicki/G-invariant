import numpy as np
from scipy.io import loadmat

path = "../../mpii/human_pose.mat"
mat = loadmat(path)
g = mat['RELEASE'][0, 0]
joints = g[0][0]
split = g[1][0]
activity = g[4]
x = []
y = []
sp = []
for i in range(len(joints)):
#for i in range(10):
    a = activity[i, 0]
    a_name = a[0]
    if a_name:
        a_name = a_name[0]
        j = joints[i][1]
        if len(j):
            try:
                for k in range(len(j[0, 0][4][0])):
                    j1 = list(j[0, 0][4][0, k][0][0])
                    j1 = [(jt[0][0, 0], jt[1][0, 0], jt[2][0, 0]) for jt in j1]
                    j1.sort(key=lambda x: x[2])
                    j1 = np.array(j1, dtype=np.float32)[:, :-1]
                    if len(j1) == 16:
                        x.append(j1)
                        y.append(a_name)
                        sp.append(split[i])
            except IndexError:
                continue
x = np.stack(x, axis=0)
y = np.array(y)
sp = np.array(sp)
np.save("x1.npy", x)
np.save("y1.npy", y)
np.save("split1.npy", sp)

