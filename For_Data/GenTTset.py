# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:27:06 2019

@author: agate
"""

import numpy as np

# filename = 'Record_para_rate_trial110000_test_7d_XYonly_original7d_20190215-144741.npz'
ext = "NeuralData_3X_1110/"
filename = "Record_Neu300_input7d_IDDoutput2d__trial60000_3X_20191108-110520.npz"

container = np.load(ext + filename)
X, Y = [container[key] for key in container]
# Y = Y[:,:2]
print(X.shape)
print(Y.shape)

num_trn = 50000
num_tst = 10000
X_trn = X[:num_trn]
Y_trn = Y[:num_trn]
np.savez(ext + "trainset_ip7d_op2d_{}_{}.npz".format(num_trn, ext[:-1]), X_trn, Y_trn)

X_tst = X[-num_tst:]
Y_tst = Y[-num_tst:]
np.savez(ext + "testset_ip7d_op2d_{}_{}.npz".format(num_tst, ext[:-1]), X_tst, Y_tst)
