# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:38:08 2020

@author: agate
"""
import sys
sys.path.append("..")
from for_figures import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
np.set_printoptions(suppress = True)
def weighted_mse(y_true,y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)

def gen_inputs(param, rg = [50,3000],num = 200):
    Inputs_E = np.linspace(rg[0],rg[1],num)
    Inputs = np.tile(param,[num,1])
    Inputs[:,5] = Inputs_E
    return Inputs

def gen_inputs2(param, rg = [50,3000],num = 200):
    Inputs_E = np.linspace(0,param[5]*1.1,num)
    Inputs = np.tile(param,[num,1])
    Inputs[:,5] = Inputs_E
    return Inputs

param = np.array([0.02889285, 2.28543134, 0.26031823, 0.51865791, 0.47421353, 224.10490706, 5.94135346])
param = np.array([  0.02910985,   1.54325873,   0.27585302,   0.80892327,
         0.39806158, 392.43782533,   4.29267559])
param = np.array([  0.02036071,   1.53052456,   0.20333969,   0.51283936,
         0.53696188, 945.97994391,   4.98856507])
param = np.array([   0.02050329,    1.51206355,    0.22756041,    0.83508421,
          0.428989  , 2476.89115275,    5.89446281])
param = np.array([   0.02021626,    1.50596259,    0.2624286 ,    0.54133358,
          0.38666466, 2475.20467823,    3.92758524])
    
param = np.array([  0.02078561,   1.5397371 ,   0.25448628,   0.98349491,
         0.39952575, 651.87970738,   5.14915522])
    
#param = np.array([  0.02148537,   2.62644008,   0.20168312,   0.57055408,
#         0.42634102, 767.08598396,   5.18342901])
    
param = np.array([   0.0204871 ,    1.50170028,    0.28622022,    0.53472142,
          0.60851262, 1001.38004977,    3.50760883])

param = np.array([   0.02926102,    1.89264095,    0.34490637,    0.98235691,    0.34299501, 2498.99184359,    3.5482721 ])
param = np.array([   0.02303655,    1.53690705,    0.44277166,    0.79246804,    0.64578675, 2584.8102844 ,    2.3757104 ])
param = np.array([   0.02988405,    1.72829303,    0.25294014,    0.94785044,    0.645256  , 2309.54305736,    5.16011527])
param = np.array([   0.02799966,    1.68842277,    0.24001271,    0.88420314,    0.34388473, 2754.97595364,    5.44822931])
param = np.array([   0.02469941,    1.55381264,    0.27178001,    0.64335198,    0.4683986 , 2666.40507201,    3.9489944 ])
param = np.array([   0.0205158 ,    1.64700134,    0.33634163,    0.58910851,    0.41672877, 2870.01763459,    2.90326221])
param = np.array([   0.02029747,    1.69224925,    0.24087275,    0.97649716,    0.5737018 , 2551.12991812,    5.79405632])
param = np.array([   0.02038053,    1.71013396,    0.25492053,    0.9861715 ,    0.42578434, 2719.3412522 ,    5.42416772])
#param = np.array([   0.02809691,    2.9545815,     0.20316503,    0.61938335,    0.66109471,
# 2233.95304708,    4.05880589])
para_test = gen_inputs2(param)

# sess = tf.Session()
# sess = tf.compat.v1.Session()
# K.set_session(sess)

name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850.h5'
model = load_model(name,custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})

# hess0 = tf.hessians(model.output[:,0],model.input)[0]
# hess1 = tf.hessians(model.output[:,1],model.input)[0]
# grads0 = K.gradients(model.output[:,0],model.input)[0]
# grads1 = K.gradients(model.output[:,1],model.input)[0]

# opi,gd0,gd1,hs0,hs1 = sess.run([model.output,grads0,grads1,hess0,hess1],feed_dict={model.input:param[np.newaxis,:]})

op_pred = model.predict(para_test)

call_pltsettings(scale_dpi = 1,scale = 1.5,fontscale = 1.9,ratio = [1,1])
plt.figure()
plt.plot(para_test[:,5],op_pred)
# plt.title(param)
# plt.show()
plt.savefig('tmp.png')

