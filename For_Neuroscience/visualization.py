import matplotlib.pyplot as plt
from Proj_Lib.for_figures import *


def showST(ST, N_E=225, lim=(0, 3)):
    s_mon_t = ST[0]
    s_mon_i = ST[1]
    call_pltsettings(scale_dpi=0.5)
    plt.figure()
    plt.plot(s_mon_t[s_mon_i < N_E], s_mon_i[s_mon_i < N_E], ".r", markersize=4)
    plt.plot(s_mon_t[s_mon_i >= N_E], s_mon_i[s_mon_i >= N_E], ".b", markersize=4)
    plt.xlim(lim)
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron index")
    plt.show()
