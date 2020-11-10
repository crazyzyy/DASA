import numpy as np
import matplotlib.pyplot as plt


def PSD_ST(ST, dt=2.5e-3, cut=1, Tlen=2, NeuNum=300, window=0.2, silent=1):
    #
    Tseq = ST[0]
    Nseq = ST[1]
    ind_eff = Tseq > cut
    Tseq = Tseq[ind_eff] - cut
    Nseq = Nseq[ind_eff]
    num_bin = int(Tlen / dt)
    move_size = int(window / dt)

    bin_ax = np.arange(num_bin + 1) * dt
    bincount = np.histogram(Tseq, bins=bin_ax)[0]

    ratebin = bincount / dt / NeuNum

    ratebin_MA = np.array(
        [ratebin[i : i + move_size] for i in range(num_bin - move_size)]
    )
    ratebinFT = 1 / np.sqrt(Tlen) * np.fft.fft(ratebin_MA * dt, axis=1)
    PSD = np.mean(np.abs(ratebinFT) ** 2, axis=0)
    PSD_half = PSD[: int(move_size / 2)]
    freqind = 1 / window * np.arange(int(move_size / 2))
    if silent == 0:
        plt.figure()
        plt.plot(freqind, PSD_half, "*-")
        plt.show()
    return [freqind, PSD_half]


def peak_PSD(PSD0):
    freq, PSD = PSD0
    base = PSD[0]
    peak_ind = [False] + list((PSD[1:-1] > PSD[:-2]) & (PSD[1:-1] > PSD[2:])) + [True]
    peak_s = PSD[peak_ind]
    peak_f = freq[peak_ind]
    p_id = np.argmax(peak_s)
    p_s = peak_s[p_id]
    p_f = peak_f[p_id]
    ratio = p_s / base
    return [p_f, p_s, ratio]


flist_default = np.arange(10, 130, 10)


def retrive_flist(PSD0, flist=flist_default, window=0.2):
    res = 1 / window
    fl_ind = (flist / res).astype(int)
    freq, PSD = PSD0
    return PSD[fl_ind]


def PSD_analysis(ST_all):
    PSD_flist_all = []
    for ST in ST_all:
        PSD0 = PSD_ST(ST)
        PSD_flist = retrive_flist(PSD0)
        PSD_flist_all.append(PSD_flist)
    return np.asarray(PSD_flist_all)
