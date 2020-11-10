# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:07:59 2019

@author: agate
"""

import h5py
import json

file_name = 'Models\DNN_800_200_sigmoid_Adam_Tsample10000_Epoch1000_X7d_Y2d__20191111-162117.h5'
f1 = h5py.File(file_name, 'r+')     # open the file
model_config0 = f1.attrs.get('model_config')
model_config = json.loads(model_config0.decode('utf-8'))
[x for x in model_config['config']['layers'][1]['config']['function']]
entry = model_config['config']['layers'][1]['config']['function'][2]

tc0 = f1.attrs.get('training_config')
tc0 = json.loads(tc0.decode('utf-8'))


model_config['config']['layers'][1]['config']['function'][2] = [xx['value'] for xx in entry]
mc2 = json.dumps(model_config).encode('utf-8')
f1.attrs['model_config'] = mc2

tc0['loss'] = 'weighted_mse'
f1.attrs['training_config'] = json.dumps(tc0).encode('utf-8')
#
f1.close()



