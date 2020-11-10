import numpy as np
import matplotlib.pyplot as plt


def SSI_compute(ST, dt=4e-3, cut=1, Tlen=2, NeuNum=300):
    Tseq = ST[0]
    Nseq = ST[1]
    ind_eff = Tseq > cut
    ddt = 1 / 16 / 1000
    bin_ax = np.linspace(1, 3, 2000 * 16 + 1)
    bincount_s = np.histogram(Tseq, bins=bin_ax)[0]
    binlen = int(dt / ddt)
    a1 = np.array([1] * binlen)
    bincount_eff = bincount_s[int(binlen / 2) : -int(binlen / 2) + 1]
    conv_s = np.convolve(a1, bincount_s, mode="valid")
    SSI = np.sum(conv_s / NeuNum * bincount_eff) / np.sum(bincount_eff)
    SSI_null = np.sum(ind_eff) / Tlen * dt / NeuNum
    SSI_ratio = SSI / SSI_null
    return [SSI, SSI_ratio]


def SSI_analysis(ST_all):
    SSI_all = []
    for ST in ST_all:
        SSI = SSI_compute(ST)
        SSI_all.append(SSI)
    return np.asarray(SSI_all)