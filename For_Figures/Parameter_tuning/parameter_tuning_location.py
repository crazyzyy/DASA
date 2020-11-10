# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 18:31:36 2019

@author: agate
"""

#parameter tuning
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from SNN_simulators import *

tgt_cand = [[5,20]]
for ii in range(len(tgt_cand)):
    tgt = tgt_cand[ii]
    print(tgt)
    rg1 = np.array([0.02,1.5,0.2,0.5,1/3,  25, 2])
    rg2 = np.array([0.03,  3,0.5,  1,2/3,3000, 6])
    
    rs = (rg2-rg1)[np.newaxis,:]
    num = 100
    para_pre = np.random.rand(num,7)
    para_cand = rg1+para_pre*(rg2-rg1)
    para_cand = rg1+0.2*(rg2-rg1)+para_pre*(rg2-rg1)*0.6
    
#    minv = np.ones([para_cand.shape[0],1])*rg1[np.newaxis,:]
#    maxv = np.ones([para_cand.shape[0],1])*rg2[np.newaxis,:]
    
    sess = tf.Session()
    K.set_session(sess)
    def weighted_mse(y_true,y_pred):
        return K.mean(K.square((y_pred - y_true)), axis=-1)
    name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850'
#    name = 'DNN_800_200_sigmoid_Adam_Tsample2500_Epoch20000_X7d_Y2d__20191125-020431_trn_ 0.0545_tst_ 0.3049'
    model = load_model(name+'.h5',
                       custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})
    
    model.trainable = False
    
    def IDD_model(model0):
        layers = [l for l in model0.layers]
        x = layers[0].output
        for i in range(1, len(layers)):
#            print(i)
            if i == 1:
                x = layers[i](x)
            elif i==2:
                out_a = layers[i](x)
            elif i==3:
                out_b = layers[i](x)
            elif i==4:
                x = layers[i]([out_a,out_b])
            else:
                x = layers[i](x)
    
        new_model = tf.keras.models.Model(inputs=layers[0].input, outputs=x)
        return new_model
    
    
    def insert_intermediate_layer_in_keras(model0,new_layer):
        layers = [l for l in model0.layers]
        x = layers[0].output
        x = new_layer(x)
        for i in range(1, len(layers)):
#            print(i)
            if i == 1:
                x = layers[i](x)
            elif i==2:
                out_a = layers[i](x)
            elif i==3:
                out_b = layers[i](x)
            elif i==4:
                x = layers[i]([out_a,out_b])
            else:
                x = layers[i](x)
    
        new_model = tf.keras.models.Model(inputs=layers[0].input, outputs=x)
        return new_model
    
    def fix_model(model0):
        for l in model0.layers:
            l.trainable=False
    
    minv = np.ones([para_cand.shape[0],1])*rg1[np.newaxis,:]/rs
    maxv = np.ones([para_cand.shape[0],1])*rg2[np.newaxis,:]/rs
    fix_model(model)    
    W0 = tf.Variable(para_cand/rs,dtype = tf.float32,trainable = True,name = 'fdasfdsa',constraint=lambda x: tf.clip_by_value(x, minv, maxv)) 
    new_layer0 = tf.keras.layers.Lambda(lambda x: x*W0)
    new_model0 = insert_intermediate_layer_in_keras(model,new_layer0)
    
                
    train_X = np.ones(para_cand.shape)*rs
    train_Y = np.ones([para_cand.shape[0],1])*np.array([tgt])
    
    loss_mse = tf.reduce_mean(tf.pow(new_model0.output-train_Y,2))
    
#    optimizer = tf.train.AdamOptimizer(0.002,beta1=0.9,beta2=0.999,epsilon=1e-08)
    optimizer = tf.train.AdamOptimizer(0.001,beta1=0.9,beta2=0.999,epsilon=1e-08)
#    optimizer = tf.train.AdamOptimizer(0.002,beta1=0.9,beta2=0.999,epsilon=1e-08)
#    optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss_mse,var_list=[W0])
    
    init_op = tf.variables_initializer(optimizer.variables())
    #sess.run(init_op)
    init_new_vars_op = tf.variables_initializer([W0])
    sess.run([init_new_vars_op,init_op])
    
    loss_all = []
    for i in range(10000):
        _,lossi = sess.run([train,loss_mse],feed_dict = {new_model0.input:train_X})
        loss_all.append(lossi)
        if lossi<1e-6:
            break
    plt.figure()
    plt.loglog(loss_all)
    plt.show()
        
    para_post, op = sess.run([W0,new_model0.output],feed_dict = {new_model0.input:train_X})
    para_post = para_post*rs

    op_pred = model.predict(para_post)
    
    import matplotlib.colors as mcolors
    plt.figure()
    plt.hist2d(op_pred[:,0],op_pred[:,1],bins = 30,cmap = 'Reds',norm=mcolors.LogNorm(1,vmax=None,clip=True))
    plt.colorbar()
    plt.show()
    
    err_l1 = np.sum(np.abs(model.predict(para_post)-train_Y),axis = 1)
    id_good = (err_l1<0.4)
    
    np.set_printoptions(suppress = True,linewidth = 125)
    print(para_post[~id_good][:100])
    print(op_pred[~id_good][:100])
    print(para_post[(op_pred[:,0]<1)&(op_pred[:,1]<1)][:100])
    print(para_post[(op_pred[:,0]<1)&(op_pred[:,1]>1)][:100])
    
    


