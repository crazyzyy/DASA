# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:03:46 2019

@author: agate
"""
#import sys
#if __name__ == "__main__":
#    ext = sys.argv[1]
#    ext2 = sys.argv[2]
ext = 'NeuralData_3X_1110'
ext2 = 'United'

import numpy as np
import glob

def loadmulti(files):
    train_X = []
    train_Y = []
    for name in files:
        container = np.load(name)
        X,Y = [container[key] for key in container]
        train_X = list(train_X)+list(X)
        train_Y = list(train_Y)+list(Y)
    return np.asarray(train_X),np.asarray(train_Y)

#files = glob.glob('./{}{}_data/*.npz'.format(ext2,ext))
files = glob.glob('D:/WeSync/Codes/Proj_NS_1/Data/{}/*.npz'.format(ext))

dataset_index = range(len(files))
train_X,train_Y = loadmulti([files[x] for x in dataset_index])
#note there might be cases when X Y are of opposite order
#train_Y = train_Y[:,:2]

print(train_X.shape)
print(train_Y.shape)

str0 = files[0]
id0 = str0.find('trial')
id1 = str0.find('_',id0)
#print(str0[id0:id1])
name = list(str0)
name[id0:id1] = '{}_trial{}'.format(ext2,train_X.shape[0])
name = ''.join(name)
#np.savez(name,train_X,train_Y)


#Complete
X = train_X
Y = train_Y
num_tst = 10000
num_trn = train_X.shape[0]-num_tst
X_trn = X[:num_trn]
Y_trn = Y[:num_trn]
X_tst = X[-num_tst:]  
Y_tst = Y[-num_tst:]

dataset = {'train': [X_trn,Y_trn],
            'test': [X_tst,Y_tst]}
np.save('Dataset_{}_trn{}_tst{}.npy'.format(ext,num_trn,num_tst),dataset)
