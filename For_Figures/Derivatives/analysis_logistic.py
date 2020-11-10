# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:00:52 2020

@author: agate
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

fig_data = np.load('hess_gd_all_20200130-121345_fullrange.npy',allow_pickle=True).item()
globals().update(fig_data)
print(fig_data.keys(),'are released into workspace!')

def nml(X):
    return (X-np.mean(X,axis = 0,keepdims = True))/np.std(X,axis = 0,keepdims = True)

def LG_anal(X,Y):
    clf0 = LogisticRegression(penalty = 'none').fit(X, Y)
    print('score: ',clf0.score(X,Y),'\n coef: ',clf0.coef_)
    clf = LogisticRegression(penalty = 'l1',solver = 'liblinear',C = 0.004).fit(X, Y)
    print('(l1) score: ',clf.score(X,Y),'\n coef: ',clf.coef_)
    return (clf0.coef_**2/np.sum(clf0.coef_**2))[0]
    
Y = np.sign(gd1[id_suit][:,6])

def conv_para(para):
    para2 = np.copy(para)
    para2[:,1] = para2[:,1]*para2[:,0]
    para2[:,2] = para2[:,2]*para2[:,0]
    para2[:,3] = para2[:,3]*para2[:,1]
    return para2




X0 = para_cand[id_suit]/(rg2-rg1)[np.newaxis,:]
# X0 = conv_para(para_cand[id_suit])
X1 = np.concatenate([X0,op_pred[id_suit]/np.max(op_pred[id_suit],axis = 0,keepdims=True)],axis = 1)

pct0 = LG_anal(nml(X0),Y)
pct1 = LG_anal(nml(X1),Y)
pct_57 = LG_anal(nml(X1[:,(5,7)]),Y)
pct_67 = LG_anal(nml(X1[:,(6,7)]),Y)

width = 0.3
plt.figure()
plt.bar(np.arange(7)+1-width/2,pct0,width,color = 'c')
plt.bar(np.arange(9)+1+width/2,pct1,width,color = 'k')
plt.show()

Data = {'pct0':pct0,'pct1':pct1}
# np.save('importance_analysis.npy',Data)

