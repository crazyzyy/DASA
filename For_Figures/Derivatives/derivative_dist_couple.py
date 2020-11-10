# derivative distribution

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def weighted_mse(y_true,y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)

sess = tf.Session()
K.set_session(sess)

nonlin_idx_all = []
np.random.seed(11)

#rg1 = np.array([0.02,1.5,0.2,0.5,1/3,  25, 2])
#rg2 = np.array([0.03,  3,0.5,  1,2/3,3000, 6])

rg1 = np.array([0.02,1.5,0.2,0.5,1/3,   25, 2])
rg2 = np.array([0.03,  3,0.5,  1,2/3, 3000, 6])

rs = (rg2-rg1)[np.newaxis,:]

#rg2[5] = 3000 
num = 10000000
#para_pre = np.random.rand(num,7)
#para_cand = rg1+para_pre*(rg2-rg1)

name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850'
model = load_model(name+'.h5',
                   custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})
 
#hess0 = tf.hessians(model.output[:,0],model.input)[0]
#hess1 = tf.hessians(model.output[:,1],model.input)[0]
grads0 = tf.gradients(model.output[:,0],model.input)[0][:,5]
hess0 = K.gradients(grads0,model.input)[0][:,5]

#opi = model.predict(para_cand)
#op_pred = np.asarray(opi)
#
#Data0 = {'para_cand':para_cand,'op_pred':op_pred}
#np.save('DNN_Prediction_1e7.npy',Data0)

Data0 = np.load('DNN_Prediction_1e7.npy',allow_pickle=True).item()   
globals().update(Data0)
print(Data0.keys(),'are released into workspace!')

id_suit = ((op_pred[:,0]>27) & (op_pred[:,0]<30))
id_suit = (id_suit & (op_pred[:,1]/op_pred[:,0]>2.5) & (op_pred[:,1]/op_pred[:,0]<5.5))

num = np.sum(id_suit)
print(num)

para_suit = para_cand[id_suit]

op_all = []
hs_all = []
ratio = np.linspace(0,1,18)[1:]
for i in range(len(ratio)):
    para_test = para_suit.copy()
    para_test[:,5] = para_test[:,5]*ratio[i]
    opi,hsi = sess.run([model.output,hess0],feed_dict={model.input:para_test})
    op_all.append(opi)
    hs_all.append(hsi)
op_all = np.asarray(op_all).swapaxes(0,1)
opE_all = op_all[:,:,0]
opI_all = op_all[:,:,1]
hs_all = np.asarray(hs_all).swapaxes(0,1)

id_real = (np.prod(((opE_all<5)+(opI_all/opE_all>2.5)*(opI_all/opE_all<5.5))>0,axis = 1)>0)
print(np.sum(id_real))

para_real = para_suit[id_real]
opE_real = opE_all[id_real]
opI_real = opI_all[id_real]
hs_real = hs_all[id_real]

sc = 0.5*para_real[:,5]**2

mask1 = (opE_real>5)*(opE_real<15)
mask2 = (opE_real>20)*(opE_real<30)

mean1 = np.sum(hs_real*mask1,axis=1)*sc/np.sum(mask1,axis = 1)
mean2 = np.sum(hs_real*mask2,axis=1)*sc/np.sum(mask2,axis = 1)

mean1 = np.max(hs_real*mask1-~mask1*1000,axis=1)*sc
mean2 = np.min(hs_real*mask2+~mask2*1000,axis=1)*sc

plt.figure()
plt.scatter(mean1,mean2)
plt.ylim(-50,100)
plt.xlim(-40,40)
plt.plot([0,50],[0,0],'-k')
plt.plot([0,0],[0,-50],'-k')
plt.xlabel('5-15 (Hz)')
plt.ylabel('20-30 (Hz)')
plt.show()

#Data = {'para_real':para_real,'opE_real':opE_real,'opI_real':opI_real,'ratio':ratio,'mean1':mean1,'mean2':mean2}
#np.save('couple_hess_dist.npy',Data)


#idd = 7322
#plt.figure()
#plt.plot(para_real[idd,5]*ratio,opE_real[idd],'-r')
#plt.plot(para_real[idd,5]*ratio,opI_real[idd],'-b')
#plt.show()










#Data = {'num':num,'para_cand':para_cand,'op_pred':op_pred,'gd0':gd0,'gd1':gd1,'gd_r':gd_r,'hess0_all':hess0_all,'hess1_all':hess1_all,
#        'id_suit':id_suit,'rg1':np.squeeze(rg1),'rg2':np.squeeze(rg2),'hess0_diag_all':hess0_diag_all,'hess1_diag_all':hess1_diag_all}
#
#np.save('hess_gd_all_{}_fullrange.npy'.format(time.strftime("%Y%m%d-%H%M%S")),Data)



