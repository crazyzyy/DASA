import sys

sys.path.append("..")
# from Proj_Lib import *

rg1 = np.array([0.02, 1.5, 0.2, 0.5, 1 / 3, 25, 1])
rg2 = np.array([0.03, 3, 0.5, 1, 2 / 3, 3000, 5])

n_trial = 11
from datetime import datetime

np.random.seed(int(str(datetime.now())[-6:]))
print(np.random.rand())
para_all = (
    rg1[np.newaxis, :] + np.random.rand(n_trial, len(rg1)) * (rg2 - rg1)[np.newaxis, :]
)
ext = "original7d"
para2rate(para_all, ext=ext)
