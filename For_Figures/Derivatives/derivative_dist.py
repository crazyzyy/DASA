# derivative distribution

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def weighted_mse(y_true,y_pred):
    return K.mean(K.square((y_pred - y_true)), axis=-1)

sess = tf.Session()
K.set_session(sess)

nonlin_idx_all = []
np.random.seed(11)

rg1 = np.array([0.02,1.5,0.2,0.5,1/3,  25, 2])
rg2 = np.array([0.03,  3,0.5,  1,2/3,3000, 6])

rs = (rg2-rg1)[np.newaxis,:]

#rg2[5] = 3000 
num = 1000000
para_pre = np.random.rand(num,7)
para_cand = rg1+para_pre*(rg2-rg1)

name = 'DNN_800_200_sigmoid_Adam_Tsample20000_Epoch20000_X7d_Y2d__20191129-032138_trn_ 0.0491_tst_ 0.0850'
model = load_model(name+'.h5',
                   custom_objects={'weighted_mse': weighted_mse,"GlorotUniform": tf.keras.initializers.glorot_uniform})
 

op_all = []
gd0_all = []
gd1_all = []
grads0 = K.gradients(model.output[:,0],model.input)[0]
grads1 = K.gradients(model.output[:,1],model.input)[0]

opi,gd0,gd1 = sess.run([model.output,grads0,grads1],feed_dict={model.input:para_cand})

op_pred = np.asarray(opi)
gd0 = np.squeeze(gd0)
gd1 = np.squeeze(gd1)

#call_pltsettings(ratio = [1,1])
plt.figure()
plt.plot(op_pred[:,0],op_pred[:,1],'k.')
plt.xlabel('pred rE')
plt.ylabel('pred rI')
plt.show()

id_suit = ((op_pred[:,0]>1) & (op_pred[:,1]>1) & (op_pred[:,1]<150) & (op_pred[:,0]<35))
id_suit = (id_suit & (op_pred[:,1]/op_pred[:,0]>2.5) & (op_pred[:,1]/op_pred[:,0]<5.5))
gd_r = (-gd0*op_pred[:,1:]/op_pred[:,0:1]+gd1)/op_pred[:,0:1]
Data = {'num':num,'op_pred':op_pred,'gd0':gd0,'gd1':gd1,'gd_r':gd_r,'id_suit':id_suit,'rg1':np.squeeze(rg1),'rg2':np.squeeze(rg2)}

np.save('derivatives.npy',Data)

num = np.sum(id_suit)
print(num)

X = (np.arange(7)+1)[np.newaxis,:]*np.ones([gd0.shape[0],1])
import matplotlib.colors as mcolors
plt.figure()
plt.hist2d(X.ravel(), (gd0*rs).ravel(), bins=[0.5+np.arange(8),200],cmap = 'Reds', norm=mcolors.LogNorm(vmin=10,vmax=None,clip=True))
plt.plot(np.arange(8)+0.5,np.zeros(8),'-k',linewidth = 3)
plt.xlabel('index')
plt.ylabel('derivative')
plt.colorbar()
plt.show()

plt.figure()
plt.hist2d(X.ravel(), (gd1*rs).ravel(), bins=[0.5+np.arange(8),200],cmap = 'Blues', norm=mcolors.LogNorm(vmin=10,vmax=None,clip=True))
plt.xlabel('index')
plt.ylabel('derivative')
plt.colorbar()
plt.show()

#suitable
plt.figure()
plt.hist2d(X[id_suit,:].ravel(), (gd0*rs)[id_suit,:].ravel(), bins=[0.5+np.arange(8),200],cmap = 'Blues', norm=mcolors.LogNorm(vmin=10,vmax=None,clip=True))
plt.xlabel('index')
plt.ylabel('derivative')
plt.colorbar()
plt.show()

plt.figure()
plt.hist2d(X[id_suit,:].ravel(), (gd1*rs)[id_suit,:].ravel(), bins=[0.5+np.arange(8),200],cmap = 'Blues', norm=mcolors.LogNorm(vmin=10,vmax=None,clip=True))
plt.xlabel('index')
plt.ylabel('derivative')
plt.colorbar()
plt.show()






#dr^E/dp5 v.s. rE
#plt.figure()
#plt.plot(op_pred[:,0],gd0[:,5],'k.')
#plt.show()
#
#def para_op5(para,alpha):
#    p = para.copy()
#    p[:,5] = p[:,5]*alpha
#    return p
#
#opi_0,gd0_0 = sess.run([model.output,grads0],feed_dict={model.input:para_op5(para_cand,0.5)})
#opi_2,gd0_2 = sess.run([model.output,grads0],feed_dict={model.input:para_op5(para_cand,1.5)})
#
#id_b = ((gd0[:,5]-gd0_0[:,5])>0)&((gd0[:,5]-gd0_2[:,5])>0)&((opi_2[:,0]<35))
#print(np.sum(id_b))
#
#para_cand_b = para_cand[id_b]
#op_d = []
#for alpha in [0.2,0.5, 0.7,1,1.3,1.8,2.3,3]:
#    op_d.append(model.predict(para_op5(para_cand_b,alpha)))
#    
#op_d = np.asarray(op_d)
#op_d5 = op_d[:,:,0]
#plt.figure()
#plt.plot(op_d5)
#plt.show()


#para = para_cand.copy()
#rE_tg = 5
#lr = 200*5
#for i in range(500):
#    opi,gd0 = sess.run([model.output,grads0],feed_dict={model.input:para})
#    loss0 = np.mean((opi[:,0]-rE_tg)**2)
#    print(i,loss0)
#    gd0 = np.asarray(gd0)[0]
#    idd = np.abs(opi[:,0]-rE_tg)>1
#    para[idd,5] = para[idd,5]-lr*(opi[idd,0]-rE_tg)*gd0[idd,5] 
#    
#plt.figure()
#plt.subplot(2,1,1)
#plt.hist(opi[:,0],100)
#plt.subplot(2,1,2)
#plt.hist(para_cand[:,5],500)
#plt.show()



plt.figure()
plt.subplot(1,2,1)
plt.scatter(op_pred[:,0], para_cand[:,5],c=gd1[:,6],cmap='rainbow',s = 2)
plt.colorbar()
plt.subplot(1,2,2)
plt.scatter(op_pred[:,0], para_cand[:,5],c=(gd1[:,6]>0),cmap='rainbow',s= 2)
plt.show() 


