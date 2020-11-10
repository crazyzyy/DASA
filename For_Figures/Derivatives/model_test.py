# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:09:29 2020

@author: agate
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from SNN_simulators import *
import time

def weighted_mse(y_true,y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)

name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850'
model = load_model(name+'.h5',
                   custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})

nm = 30

#para_test = np.array([2.04818285e-02, 1.54597016e+00, 2.43824940e-01, 6.16006173e-01,
#       5.89488129e-01, 2.45721550e+03, 4.67151454e+00])
#para_test = np.array([2.65233916e-02, 2.00512387e+00, 2.38003109e-01, 9.87984344e-01,
#       4.45669225e-01, 2.43972160e+03, 5.65098144e+00])
#para_test = np.array([[2.43500718e-02, 2.71529174e+00, 2.16530703e-01, 9.24549836e-01,
#        3.53191037e-01, 2.56940998e+03, 5.88127376e+00]])


#params = np.array([[   0.02926102,    1.89264095,    0.34490637,    0.98235691,    0.34299501, 2498.99184359,    3.5482721 ],
#[   0.02303655,    1.53690705,    0.44277166,    0.79246804,    0.64578675, 2584.8102844 ,    2.3757104 ],
#[   0.02988405,    1.72829303,    0.25294014,    0.94785044,    0.645256  , 2309.54305736,    5.16011527],
#[   0.02799966,    1.68842277,    0.24001271,    0.88420314,    0.34388473, 2754.97595364,    5.44822931],
#[   0.02469941,    1.55381264,    0.27178001,    0.64335198,    0.4683986 , 2666.40507201,    3.9489944 ],
#[   0.0205158 ,    1.64700134,    0.33634163,    0.58910851,    0.41672877, 2870.01763459,    2.90326221],
#[   0.02029747,    1.69224925,    0.24087275,    0.97649716,    0.5737018 , 2551.12991812,    5.79405632]])
#
#tt = np.array([[-10,1],
#    [2,20],
#    [-20,0],
#    [-10,40],
#    [0,60],
#    [10,80],
#    [0,0]])

# params = np.array([
#         [   0.02455104,    2.0845956 ,    0.26789466,    0.98151239,    0.63854218, 2633.89486728,    4.81833257],
#         [   0.02293638,    1.67166667,    0.39724945,    0.9295619 ,    0.5526452 , 2717.73690496,    2.98398271],
#         [   0.02245393,    1.5880676 ,    0.41330884,    0.84280593,    0.65405352, 2853.02312655,    2.85833641],
#         ])
# tt = np.array([[-11,0],[-3,10],[4,30]])

params = np.array([[   0.02038053,    1.71013396,    0.25492053,    0.9861715 ,    0.42578434, 2719.3412522 ,    5.42416772]])
tt = np.array([[2.7,-0.3]])

op_all = []
rate_all = []
for i in range(len(params)):
    para_test = params[i]
    para_test = para_test*np.ones([nm,1])
    #para_test[:,5] = para_test[:,5]*np.linspace(0.02,1.1,nm)
    para_test[:,5] = para_test[:,5]*np.linspace(0,1,nm)
    
    op = model.predict(para_test)
    rate = para2rate(para_test)
    
    op_all.append(op)
    rate_all.append(rate)
    
    plt.figure()
    plt.plot(para_test[:,5],op)
    plt.plot(para_test[:,5],rate,'--')
    plt.title(tt[i])
    plt.show()
    plt.savefig('gain_curve_{}_.png'.format(tt[i]))
    
Data_list = ['params','tt','op_all','rate_all']
Data = {key:globals()[key] for key in Data_list}

np.save('gain_curves_data_{}.npy'.format(time.strftime("%Y%m%d-%H%M%S")),Data)


for i in range(len(params)):
    para_test = params[i]
    para_test = para_test*np.ones([nm,1])
    para_test[:,5] = para_test[:,5]*np.linspace(0,1,nm)
    
    plt.figure()
    plt.plot(para_test[:,5],np.asarray(rate_all[i])[:,0],'-k')
    plt.plot(para_test[:,5],np.asarray(op_all[i])[:,0],'--b')
    # plt.title(tt[i])
    plt.ylim(0,30)
    plt.xlabel(r'$\eta^{\rm{ext},E}$')
    plt.ylabel(r'$r^{E}$')
    # plt.show()
    plt.savefig('gain_curve_{}_ylim_.png'.format(tt[i]))




