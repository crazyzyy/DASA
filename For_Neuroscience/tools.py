import numpy as np


def loadmulti(files):
    train_X = []
    train_Y = []
    for name in files:
        container = np.load(name)
        X, Y = [container[key] for key in container]
        train_X = list(train_X) + list(X)
        train_Y = list(train_Y) + list(Y)
    return np.asarray(train_X), np.asarray(train_Y)


def loadmulti_3var(files):
    train_X = []
    train_Y = []
    ST_all = []
    for name in files:
        container = np.load(name)
        X, Y, ST = [container[key] for key in container]
        train_X = list(train_X) + list(X)
        train_Y = list(train_Y) + list(Y)
        ST_all = list(ST_all) + list(ST)
    return np.asarray(train_X), np.asarray(train_Y), ST_all


def loadf(file):
    container_random = np.load(file)
    return [container_random[key] for key in container_random]


def rescaleto(X, rs_para):
    X_rs = (X - rs_para[0]) / rs_para[1]
    return X_rs


def rescaleback(X_rs, rs_para):
    X = X_rs * rs_para[1] + rs_para[0]
    return X


def rescaletoXY(X, rs_para):
    X_rs = (X[0] - rs_para[0]) / rs_para[1]
    Y_rs = (X[1] - rs_para[2]) / rs_para[3]
    return X_rs, Y_rs


def rescalebackXY(X_rs, rs_para):
    X = X_rs[0] * rs_para[1] + rs_para[0]
    Y = X_rs[1] * rs_para[3] + rs_para[2]
    return X, Y


ip_ratio0 = np.array([1, 2, 3, 4, 5])


def para2mulIP_Mul(X0, ip_ratio=ip_ratio0):
    num_ip = len(ip_ratio)
    X_mulIP = np.tile(X0, [num_ip, 1, 1])
    X_mulIP[:, :, 5] = X_mulIP[0:1, :, 5] * ip_ratio[:, np.newaxis]
    return X_mulIP


def para2mulIP_M(X0, ip_ratio=ip_ratio0):
    num_ip = len(ip_ratio)
    X_mulIP = np.tile(X0, [num_ip, 1])
    X_mulIP[:, 5] = X_mulIP[0, 5] * np.array(ip_ratio)
    return X_mulIP


def para2mulIP_extend(params, ip_ratio=ip_ratio0):
    return (
        para2mulIP_Mul(params, ip_ratio=ip_ratio)
        .swapaxes(0, 1)
        .reshape([-1, params.shape[1]])
    )


def distXY(X, Y):
    dim = X.shape[1]
    n_test = X.shape[0]
    n_ref = Y.shape[0]
    distmat = np.sqrt(
        np.sum((X[:, :, np.newaxis] - Y.swapaxes(0, 1)[np.newaxis, :, :]) ** 2, axis=1)
    )
    return distmat


def meanXY(X, Y):
    dim = X.shape[1]
    n_test = X.shape[0]
    n_ref = Y.shape[0]
    mn = Y.swapaxes(0, 1)[np.newaxis, :, :] / 2 + X[:, :, np.newaxis] / 2
    return mn.swapaxes(1, 2)


def vs_para_distr(param, rg):
    rg1, rg2 = rg
    param_nml = (param - rg1[np.newaxis, :]) / (rg2 - rg1)[np.newaxis, :]
    mn = np.mean(param_nml, axis=0)
    std = np.std(param_nml, axis=0)
    call_pltsettings()
    plt.figure()
    plt.errorbar(np.arange(len(rg1)), mn, std)
    plt.plot(np.arange(len(rg1)), param_nml.transpose(), "g*")
    ms = 13
    plt.plot(np.arange(len(rg1)), param_nml[0], "ko", ms=ms)
    plt.plot(np.arange(len(rg1)), param_nml[1], "ro", ms=ms)
    plt.plot(np.arange(len(rg1)), param_nml[2], "bo", ms=ms)
    fs = 18
    [plt.text(i, 0, "{:.3g}".format(rg1[i]), fontsize=fs) for i in range(len(rg1))]
    [plt.text(i, 1, "{:.3g}".format(rg2[i]), fontsize=fs) for i in range(len(rg1))]
    plt.show()


def conVC_ind(x, option=0):
    L = np.shape(x)[1]
    dif_ed = x[:, -1:] - x[:, 0:1]
    lin_interp = x[:, 0:1] + np.linspace(0, 1, L)[np.newaxis, :] * dif_ed
    ratio_nonlin = (x - lin_interp) / dif_ed
    if option == 0:
        value = np.abs(np.sum(ratio_nonlin, axis=1))
    elif option == 1:
        value = -np.sum(ratio_nonlin, axis=1)  # convex
    elif option == 2:
        value = np.sum(ratio_nonlin, axis=1)  # concave
    return value


def nonlin_ind(x, mid=5):
    v1 = conVC_ind(x[:, :mid], option=1)
    v2 = conVC_ind(x[:, mid - 1 :], option=2)
    return v1 * v2 * ((v1 > 0) | (v2 > 0))
