# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 20:16:29 2019

@author: agate
"""

import numpy as np
import re
import glob
filenames = glob.glob('*.npy')

tgt_all = []
ratio_all = []
err_all = []
pred_exp_err_all = []
for name in filenames:
    data = np.load(name,allow_pickle = True).item()
    idx_ = [a.end() for a in list(re.finditer('_', name))]+[-3]
    if len(idx_) == 6:
        ratio_all.append(int(name[idx_[1]:idx_[2]-1])/100)
        tgt_all.append([int(name[idx_[j]:idx_[j+1]-1]) for j in [3,4]])
        pred_exp_err_all.append(np.asarray(data['rate_exp']))
#    err_all.append(np.sum(np.mean(np.abs(pred_exp_err),axis = 0)))
#    err_l1 = np.mean(np.abs(pred_exp_err),axis = 0)
#    err_l2 = np.sqrt(np.mean(np.abs(pred_exp_err)**2,axis = 0))
#    print(err_l1,err_l2)
    
tgt_all = np.asarray(tgt_all)
ratio_all = np.asarray(ratio_all)
err_all = np.asarray(err_all)

np.save('APT_error.npy',{'tgt_all':tgt_all,'ratio_all':ratio_all,'err_all':err_all,'pred_exp_err_all':pred_exp_err_all})