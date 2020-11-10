from .packages import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
rng = np.random

from .train_DNN import *
