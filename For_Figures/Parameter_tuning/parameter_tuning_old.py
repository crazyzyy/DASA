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

tgt_cand = [[1,5],[5,10],[5,20],[5,30],[8,16],[10,25],[10,60],[15,45],[20,10],[20,20],[20,60],[30,45],[30,100]]
for ii in range(len(tgt_cand)):
    tgt = tgt_cand[ii]
    print(tgt)
    rg1 = np.array([0.02,1.5,0.2,0.5,1/3,  25, 2])
    rg2 = np.array([0.03,  3,0.5,  1,2/3,3000, 6])
    
    rs = (rg2-rg1)[np.newaxis,:]
    num = 500
    para_pre = np.random.rand(num,7)
    #para_cand = rg1+para_pre*(rg2-rg1)
    para_cand = rg1+0.2*(rg2-rg1)+para_pre*(rg2-rg1)*0.6
    
    minv = np.ones([para_cand.shape[0],1])*rg1[np.newaxis,:]
    maxv = np.ones([para_cand.shape[0],1])*rg2[np.newaxis,:]
    
    sess = tf.Session()
    K.set_session(sess)
    def weighted_mse(y_true,y_pred):
        return K.mean(K.square((y_pred - y_true)), axis=-1)
    #name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850'
    name = 'DNN_800_200_sigmoid_Adam_Tsample2500_Epoch20000_X7d_Y2d__20191125-020431_trn_ 0.0545_tst_ 0.3049'
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
    
    fix_model(model)    
    W0 = tf.Variable(para_cand,dtype = tf.float32,trainable = True,name = 'fdasfdsa',constraint=lambda x: tf.clip_by_value(x, minv, maxv)) 
    new_layer0 = tf.keras.layers.Lambda(lambda x: x*W0)
    new_model0 = insert_intermediate_layer_in_keras(model,new_layer0)
    
                
    train_X = np.ones(para_cand.shape)
    train_Y = np.ones([para_cand.shape[0],1])*np.array([tgt])
    
    loss_mse = tf.reduce_mean(tf.pow(new_model0.output-train_Y,2))
    
    optimizer = tf.train.AdamOptimizer(0.002,beta1=0.9,beta2=0.999,epsilon=1e-08)
    #optimizer = tf.train.GradientDescentOptimizer(0.005)
    train = optimizer.minimize(loss_mse,var_list=[W0])
    
    init_op = tf.variables_initializer(optimizer.variables())
    #sess.run(init_op)
    init_new_vars_op = tf.variables_initializer([W0])
    sess.run([init_new_vars_op,init_op])
    
    for i in range(5000):
        _,lossi = sess.run([train,loss_mse],feed_dict = {new_model0.input:train_X})
#        print(lossi)
        
    para_post, op = sess.run([W0,new_model0.output],feed_dict = {new_model0.input:train_X})


    err_l1 = np.sum(np.abs(model.predict(para_post)-train_Y),axis = 1)
    id_good = (err_l1<0.4)
    print(np.sum(id_good))
    if np.sum(id_good) == 0:
        continue
    
    para_good = para_post[id_good]
    rate_exp = para2rate(para_good)
    rate_pred = model.predict(para_good)
    
    #id_suit = (np.sum(((para_good/rg1[np.newaxis,:])>=1.05)&((rg2[np.newaxis,:]/para_good)>=1.05),axis = 1)==7)
    #pred_tgt_err = rate_pred - [[10,30]]
    #pred_exp_err = rate_pred - rate_exp
    #pred_exp_suit_err = pred_exp_err[id_suit]
    #diversity = np.std(para_good,axis = 0)/(rg2-rg1)
    
    #plt.figure();plt.hist(para_good[id_suit][:,2],50);plt.show()
    #plt.figure();plt.hist(para_good[:,3],50);plt.show()
    
    np.save('APT_results_{}_tgt_{}_{}.npy'.format(np.sum(id_good),tgt[0],tgt[1]),{'para_good':para_good,'rate_exp':rate_exp,'rate_pred':rate_pred})
    
    #model = load_model(name+'.h5',
    #                   custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})
    #model.predict(para_post[0:10])

