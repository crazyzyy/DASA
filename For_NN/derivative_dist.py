# derivative distribution

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def weighted_mse(y_true, y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)


sess = tf.Session()
K.set_session(sess)

nonlin_idx_all = []
np.random.seed(11)


rg1 = np.array([[0.02, 1.5, 0.2, 0.5, 1 / 3, 25, 2]])
rg2 = np.array([[0.03, 3, 0.5, 1, 2 / 3, 3000, 6]])

# rg1 = np.array([[0.001, 0.05,0.05,0.1,0.01,   5,  0.1]])
# rg2 = np.array([[0.2,     10,   5,  5,   2,10000,  10]])

rs = (rg2 - rg1)[np.newaxis, :]

num = 200000
para_pre = np.random.rand(num, 7)
para_cand = rg1 + para_pre * (rg2 - rg1)

name = "DNN_800_200_sigmoid_Adam_Tsample10000_Epoch2000_X7d_Y2d__20191114-191204"
# name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch5000_X7d_Y2d__20191117-014614'
name = "Models\\" + name
model = load_model(
    name + ".h5",
    custom_objects={
        "weighted_mse": weighted_mse,
        "GlorotUniform": tf.keras.initializers.glorot_uniform,
    },
)


op_all = []
gd0_all = []
gd1_all = []
grads0 = K.gradients(model.output[:, 0], model.input)[0]
grads1 = K.gradients(model.output[:, 1], model.input)[0]

opi, gd0, gd1 = sess.run(
    [model.output, grads0, grads1], feed_dict={model.input: para_cand}
)
# gd0 = np.asarray(gd0)[0]#.swapaxes(0,1)
# gd1 = np.asarray(gd1)[0]#.swapaxes(0,1)\

# op_all = np.asarray(op_all).swapaxes(0,1)
op_pred = np.asarray(opi)
# call_pltsettings(ratio = [1,1])
plt.figure()
plt.plot(op_pred[:, 0], op_pred[:, 1], "k.")
plt.xlabel("pred rE")
plt.ylabel("pred rI")
plt.show()

id_suit = (op_pred[:, 0] > 1) & (op_pred[:, 1] > 1)
id_suit = id_suit & (op_pred[:, 0] > 5) & (op_pred[:, 0] < 30)
id_suit = (
    id_suit
    & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
    & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
)

num = np.sum(id_suit)
print(num)

plt.figure()
plt.plot(op_pred[:, 0], op_pred[:, 1], "k.")
plt.plot(op_pred[id_suit][:, 0], op_pred[id_suit][:, 1], "r.")
plt.xlabel("pred rE")
plt.ylabel("pred rI")
plt.show()


gd0 = gd0[id_suit]
gd1 = gd1[id_suit]

para_pre = para_pre[id_suit]
para_cand = para_cand[id_suit]
op_pred = op_pred[id_suit]

gd_r = (-gd0 * op_pred[:, 1:] / op_pred[:, 0:1] + gd1) / op_pred[:, 0:1]


X_spread = (np.arange(7) + 1)[np.newaxis, :] + np.random.randn(num, 7) * 0.1
plt.figure()
plt.subplot(131)
plt.scatter(X_spread, gd0 * rs, c="b", s=3)
plt.plot(np.arange(7) + 1, np.zeros(7), "-k")
plt.plot(np.arange(7) + 1, np.mean(np.abs(gd0 * rs)[0], axis=0), "-r")
plt.title("dr^E/dp_i (IO normalized)")
plt.subplot(132)
plt.scatter(X_spread, gd1 * rs, c="b", s=3)
plt.plot(np.arange(7) + 1, np.zeros(7), "-k")
plt.plot(np.arange(7) + 1, np.mean(np.abs(gd1 * rs)[0], axis=0), "-r")
plt.title("dr^I/dp_i (IO normalized)")
plt.subplot(133)
plt.scatter(X_spread, gd_r * rs, c="b", s=3)
plt.plot(np.arange(7) + 1, np.zeros(7), "-k")
plt.plot(np.arange(7) + 1, np.mean(np.abs(gd_r * rs)[0], axis=0), "-r")
plt.title("d(r^I/r^E)/dp_i (IO normalized)")
plt.show()


plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(op_pred[:, 0], para_cand[:, 5], c=gd1[:, 6], cmap="rainbow", s=2)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.scatter(op_pred[:, 0], para_cand[:, 5], c=(gd1[:, 6] > 0), cmap="rainbow", s=2)
plt.show()
