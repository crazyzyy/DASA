import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense,Subtract,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import time
from tensorflow.keras.utils import get_custom_objects


def DNN_train(dataset,ASI = True,ActFun = 'relu',layers = [800,200,200],lr = 5e-5, \
            EPOCHS = 400,OptMethod = 'Adam',tnum = -1):
#    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    f_Y = "IDD"
    train_X,train_Y = dataset['train']
#    tnum = 2000
    train_X = train_X[:tnum,:]
    train_Y = train_Y[:tnum,:]
    test_X,test_Y = dataset['test']
    n_samples,n_input = train_X.shape
    n_output = train_Y.shape[1]
    rescale_para = [train_X.mean(axis=0),train_X.std(axis=0), \
                train_Y.mean(axis=0),train_Y.std(axis=0)]
    [mean,std,mean2,std2] = rescale_para
    rescale_para_list = [list(xx) for xx in rescale_para]

    model = build_model_normalized_ASI(n_input = n_input,n_output = n_output,
            rescale_para = rescale_para_list, base_model = build_model_dense,
            layers = layers,ActFun = ActFun,ASI = ASI)
    

#    wmse = lambda y_true,y_pred:weighted_mse(y_true,y_pred,weight = std2)
#    weight = std2
    def weighted_mse(y_true,y_pred):
        weight = tf.expand_dims(tf.convert_to_tensor(std2,tf.float32),0)
        return K.mean(K.square((y_pred - y_true)/weight), axis=-1)
    
    get_custom_objects().update({"weighted_mse": weighted_mse})
    
    optimizer = keras.optimizers.__dict__[OptMethod](lr=lr)
    model.compile(loss=weighted_mse,optimizer=optimizer,metrics=['mae'])
#    model.summary()

    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 50 == 0:
#            print('')
            print('.', end='')

    # Store training stats
    vs = 0.0
    history = model.fit(train_X, train_Y,# batch_size=500,
                        epochs=EPOCHS,
#                        validation_split=vs, 
                        verbose=0,
                        callbacks=[PrintDot()])
#    model.save('tmp'+'.h5')
    name = 'DNN_800_200_{}_{}_Tsample{}_Epoch{}_X{}d_Y{}d_{}_'.format(ActFun,OptMethod,train_X.shape[0],EPOCHS,train_X.shape[1],train_Y.shape[1],str(f_Y)[10:13])+time.strftime("%Y%m%d-%H%M%S")
#    try:
#        name = 'DNN_800_200_{}_{}_Tsample{}_Epoch{}_X{}d_Y{}d_{}_'.format(ActFun,OptMethod,train_X.shape[0],EPOCHS,train_X.shape[1],train_Y.shape[1],str(f_Y)[10:13])+time.strftime("%Y%m%d-%H%M%S")
#        model.save(name+'.h5')
##        np.savez(name,rescale_para)
#    except:
#        print('cannot save model!')

    def plot_history(history):
      plt.figure()
      plt.xlabel('Epoch')
      plt.ylabel('Mean Abs Error [1000$]')
      plt.semilogy(history.epoch, np.array(history.history['mean_absolute_error']),
               label='Train Loss')
      if vs>0:
          plt.semilogy(history.epoch, np.array(history.history['val_mean_absolute_error']),
               label = 'Val loss')
      plt.legend()
#      plt.ylim([0, 5])

    plot_history(history)

    print('-----------------------------------------------------')
    [loss_tr, mae_tr] = model.evaluate(train_X, train_Y, verbose=0)
    print("Training set Mean Abs Error: {:7.4f}, loss: {:7.4f}".format(mae_tr,np.sqrt(loss_tr)))
    [loss, mae] = model.evaluate(test_X, test_Y, verbose=0)
    print("Testing set Mean Abs Error: {:7.4f}, loss: {:7.4f}".format(mae,np.sqrt(loss)))
    print('------------------------------------------------------')
    
    model.save('Models/'+name+'_trn_{:7.4f}_tst_{:7.4f}.h5'.format(mae_tr,mae))
    
    test_Y_pd = model.predict(test_X)
    np.set_printoptions(suppress=True,linewidth=150)
    diff =  test_Y_pd-test_Y
    absdif = np.mean(np.abs(diff),axis = 0)
    stddif = np.std(diff,axis = 0)
    print('test meanabs err:{} std:{}'.format(absdif,stddif))
    print('residual/Y: meanabs: {} std: {}'.format(absdif/np.mean(np.abs(test_Y),axis = 0),stddif/np.std(test_Y,axis = 0)))
    print('uncertainty reduction: {}'.format(1-stddif**2/np.var(test_Y,axis = 0)))
    return model, rescale_para

def build_model_dense(n_input,n_output,layers = [800,200,200],ActFun = 'relu'):
    model_input = Input(shape = (n_input,))
    ini = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    x = model_input
    for num in layers:
        x = Dense(num, activation=tf.nn.__dict__[ActFun],bias_initializer=ini, \
                  kernel_initializer=ini)(x)
    model_out = Dense(n_output)(x)
    return Model(model_input,model_out)

def build_model_normalized_ASI(n_input,n_output,rescale_para,base_model = build_model_dense, \
        layers = [800,200,200],ActFun = 'relu',ASI = True):
    [mean,std,mean2,std2] = rescale_para
#    def linear_transform(x):
#        v1 = K.constant(mean,dtype = tf.float32)
#        v2 = K.constant(std, dtype = tf.float32)
#        return (x-v1)/v2
    
    data_input = Input(shape=(n_input,))
    data_input_post = Lambda(lambda x: (x-mean)/std)(data_input)
#    data_input_post = Lambda(linear_transform)(data_input)
    m1 = base_model(n_input = n_input,n_output = n_output,layers = layers,ActFun = ActFun)
    m2 = base_model(n_input = n_input,n_output = n_output,layers = layers,ActFun = ActFun)
    if ASI == True:
        m2.set_weights(m1.get_weights())
    out_a = m1(data_input_post)
    out_b = m2(data_input_post)
    data_out_pre = Subtract()([out_a,out_b])
    data_out_pre = Lambda(lambda x:0.70710678*x)(data_out_pre)
    data_out = Lambda(lambda x:x)(data_out_pre)
#    data_out = Lambda(lambda x: x*std2+mean2)(data_out_pre)
    return Model(data_input,data_out)

def dataset_fortest():
    train_X = np.random.randn(20,7)
    test_X = np.random.randn(100,7)
    dataset_fortest = {'train': [train_X,np.mean(train_X,axis = -1,keepdims = True)], \
                'test': [test_X,np.mean(test_X,axis = -1,keepdims = True)]}
    return dataset_fortest

#from Proj_Lib import *
#DNN_train(dataset_fortest())

dataset0 = np.load('D:\WeSync\Codes\Proj_NS_1\Data\dataset_a.npy',allow_pickle=True).item()
dataset1 = np.load('D:\WeSync\Codes\Proj_NS_1\Data\Dataset_NeuralData_3X_1110_trn50000_tst10000.npy',allow_pickle=True).item()
dataset2 = np.load('D:\WeSync\Codes\Proj_NS_1\Data\Dataset_NeuralData_10X_1110_trn70000_tst10000.npy',allow_pickle=True).item()

dataset = dataset1
#DNN_train(dataset,ASI = True,ActFun = 'tanh',layers = [1000,100],lr = 1e-5,EPOCHS = 2000,OptMethod = 'Adam');
#DNN_train(dataset,ASI = True,ActFun = 'relu',layers = [800,200,200],lr = 5e-5,EPOCHS = 800,OptMethod = 'Adam');
#DNN_train(dataset,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 5e-5,EPOCHS=3000)
#DNN_train(dataset,ASI = True,ActFun = 'tanh',layers = [800,200,200],lr = 3e-5,EPOCHS=1000)

#for tnum in [200, 500, 1250,2500,5000,10000,20000,40000]:
##for tnum in [10000,20000,40000]:
##for tnum in [500, 1000, 2000, 5000,10000]:
##    DNN_train(dataset1,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 3e-5,EPOCHS=5000,tnum = tnum)
##    DNN_train(dataset1,ASI = True,ActFun = 'tanh',layers = [800,200,200],lr = 2e-4,EPOCHS=1000,tnum = tnum)
##    DNN_train(dataset,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 1.5e-3/np.sqrt(tnum),EPOCHS=7500,tnum = tnum)
#    DNN_train(dataset,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 2.5e-3/np.sqrt(tnum),EPOCHS=10000,tnum = tnum)
##    DNN_train(dataset1,ASI = True,ActFun = 'relu',layers = [800,200,200],lr = 5e-5,EPOCHS=3000,tnum = tnum)
##DNN_train(dataset,ASI = True,ActFun = 'relu',layers = [800,200,200],lr = 3e-5,EPOCHS=1000,tnum = 1000)
     
dataset = dataset2   
#for tnum in [200, 500, 1250,2500,5000,10000,20000]:
for tnum in [40000]:
#    DNN_train(dataset,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 2.5e-3/np.sqrt(tnum),EPOCHS=10000,tnum = tnum)
    DNN_train(dataset,ASI = True,ActFun = 'sigmoid',layers = [800,200,200],lr = 7e-5,EPOCHS=10000,tnum = tnum)