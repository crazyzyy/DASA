import numpy as np

def loadf_2list(file):
	# load npz as a list #
    container_random = np.load(file)
    return [container_random[key] for key in container_random]

#operations for output Y
def IDD(Y):
    return Y

def LOG(Y):
    if np.ndim(Y) == 1:
        Y = Y[np.newaxis,:]
    return np.array([list(x[:2])+list(np.log(x[2:14]))+list(x[14:]) for x in Y])

def LGnmlR(Y):
    if np.ndim(Y) == 1:
        Y = Y[np.newaxis,:]
    rate = Y[:,0]*0.75+Y[:,1]*0.25
    Y[:,2:14] = np.log(Y[:,2:14]/rate[:,np.newaxis])
    return Y

def nmlR(Y):
    if np.ndim(Y) == 1:
        Y = Y[np.newaxis,:]
    rate = Y[:,0]*0.75+Y[:,1]*0.25
    Y[:,2:14] = Y[:,2:14]/rate[:,np.newaxis]
    return Y
