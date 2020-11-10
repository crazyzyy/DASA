# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:52:01 2019

@author: agate
"""

#Load Model for analysis
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense,Subtract,Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import time

def model_on_data(model,dataset,num_limit = 20000):
    train_X,train_Y = dataset['train']
    test_X,test_Y = dataset['test']
    
#    num_limit = 20000
    if train_X.shape[0]>num_limit:
        train_X = train_X[:num_limit]
        train_Y = train_Y[:num_limit]
    
    rg1 = np.array([[0.02,1.5,0.2,0.5,1/3,  25, 2]])
    rg2 = np.array([[0.03,  3,0.5,  1,2/3,3000, 6]])
    f_c = np.prod(((train_X-rg1)>=0)&((train_X-rg2)<=0),axis = 1)
    print('{} in {} are in feasible region.'.format(np.sum(f_c),train_X.shape[0]))
    
    f_c_y1 = ((train_Y[:,0]-3)>=0)&((train_Y[:,0]-30)<=0)
    f_c_y2 = ((train_Y[:,1]/(train_Y[:,0]+1e-8)-2)>=0)&((train_Y[:,1]/(train_Y[:,0]+1e-8)-6)<=0)
    f_num = np.sum(f_c_y1*f_c_y2)
    print('{} in {},ratio {} produce feasible output.'.format(f_num,train_X.shape[0],f_num/train_X.shape[0]))
    
    plt.figure()
    for i in [1,2]:
        plt.subplot(2,1,i)
        plt.hist(train_Y[:,i-1],200)
    
    Y0 = model.predict(train_X)
    Y1 = model.predict(test_X)
    print(np.std(train_Y,axis = 0))
    err0 = Y0-train_Y
    err1 = Y1-test_Y
    print_err(err0,'train')
    print_err(err1,'test')
    return err1
    
def print_err(err,str0 = ''):
    num = err.shape[0]
    absdif = np.mean(np.abs(err),axis = 0)
    stddif = np.std(err,axis = 0)
    print('{}. {} meanabs err:{} std:{}'.format(num,str0,absdif,stddif))


#notice!! it is a decoy loss simply to avoid error in loading
def weighted_mse(y_true,y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)


import glob
filenames = glob.glob('*.h5')

#dataset = np.load('D:\WeSync\Codes\Proj_NS_1\Data\dataset_a.npy',allow_pickle = True).item()
dataset = np.load('D:\WeSync\Codes\Proj_NS_1\Data\Dataset_NeuralData_10X_1110_trn70000_tst10000.npy',allow_pickle = True).item()

Tsamples = []
errors = []
for name in filenames:
    id0 = name.find('Tsample')+7
    id1 = name.find('_',id0)
    Tsamples.append(int(name[id0:id1]))
Tsamples = np.asarray(Tsamples)

idx = np.argsort(Tsamples)
Tsamples = Tsamples[idx]

filenames = [filenames[idd] for idd in idx]
for name in filenames:
    model = load_model(name,
                   custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})
    errors.append(model_on_data(model,dataset,num_limit = Tsamples[-1]).squeeze())
errors = np.asarray(errors)

err_l2 = np.linalg.norm(errors,axis = 1)/np.sqrt(errors.shape[1])
err_l1 = np.mean(np.abs(errors),axis = 1)

dict0 = {'Tsamples':Tsamples,'err_l1':err_l1,'err_l2':err_l2}
np.save('model_error_l1_l2_test10000_10X_10X.npy',dict0)

    
  
    
    
    



