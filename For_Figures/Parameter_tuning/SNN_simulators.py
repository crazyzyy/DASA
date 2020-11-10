import numpy as np
import time
from brian2 import *

def IDD(Y):
    return Y

def para2rate(para_all = np.array([0.025,2.25,0.35,0.75,0.5,500,4]),seed0 = 11,f_Y = IDD,ext = ''):
    if np.ndim(para_all) == 1:
        para_all = para_all[np.newaxis,:]
    l_para = para_all.shape[1]
    device.reinit()
    device.activate()
    n_trial = para_all.shape[0]
    Record_rate = []
    Dt = 1/16*ms

    ST_all = []
    for ii in range(n_trial):
        t0 = time.time()
        para = para_all[ii]
        SEE = para[0] #(0.02, 0.03)
        SEI = SEE*para[1]  #(1.5, 3)*SEE
        SIE = SEE*para[2] #(0.2,0.5)*SEE
        SII = SEI*para[3]  #(0.5, 1)*SEI

        alpha = para[4]*1200*second**-1 #(1/3,2/3)
        lbd_E = para[5]*Hz
        lbd_I = lbd_E*para[6]
        if len(para)==11:
            tau_E = para[7]*ms
            tau_I = para[8]*ms
            d_E   = para[9]*ms
            d_I   = para[10]*ms
        elif len(para)==7:
            tau_E = 2*ms
            tau_I = 3*ms
            d_E   = 0*ms
            d_I   = 0*ms

        N_E = 225
        N_I = 75

        tau_l = 20*ms
        V_E = 14/3
        V_I = -2/3

        Vt = 1
        Vr = 0

        eqs = '''
        dv/dt  = -v/tau_l-(v-V_E)*ge-(v-V_I)*gi : 1 (unless refractory)
        dge/dt = -ge/tau_E : second**-1
        dgi/dt = -gi/tau_I : second**-1
        '''
        P_Am = PoissonGroup(N_E+N_I, rates= alpha,dt=Dt)
        P2Ex = PoissonGroup(N_E, rates= lbd_E,dt = Dt)
        P2In = PoissonGroup(N_I, rates = lbd_I,dt=Dt)

        P = NeuronGroup(N_E+N_I, eqs, threshold='v>Vt', reset='v = Vr', refractory=2.5*ms,
                        method='euler', dt = Dt)
        P.v = 'Vr + rand() * (Vt - Vr)'
        P.ge = 0*ms**-1
        P.gi = 0*ms**-1

        ###connectivity
        C_Am = Synapses(P_Am, P, on_pre='ge += 0.005/tau_E ',dt = Dt)
        C2Ex = Synapses(P2Ex, P, on_pre='ge += SEE/tau_E ',dt = Dt)
        C2In = Synapses(P2In, P, on_pre='ge += SIE/tau_E ',dt = Dt)
        C_Am.connect(j = 'i')
        C2Ex.connect(j = 'i')
        C2In.connect(j = 'i+N_E')


        ##########################
        P2Pe = Synapses(P, P, 'we: 1',on_pre='ge += we/tau_E',dt = Dt)
        P2Pi = Synapses(P, P, 'wi: 1',on_pre='gi += wi/tau_I',dt = Dt)

        seed(seed0)
        P2Pe.connect('i<N_E and j<N_E',p=0.1)
        P2Pe.connect('i<N_E and j>=N_E',p=0.5)
        P2Pi.connect('i>=N_E and j<N_E',p=0.5)
        P2Pi.connect('i>=N_E and j>=N_E',p=0.5)
        #synaptic delay
        P2Pe.delay = 'd_E*(j<N_E)*(i<N_E)'
        P2Pi.delay = 'd_I*(j<N_E)*(i>=N_E)'

        P2Pe.we = 'SEE*(j<N_E)*(0.8+0.2*rand())+SIE*(j>=N_E)'
        P2Pi.wi = 'SEI*(j<N_E)+SII*(j>=N_E)'
        #################################
        s_mon = SpikeMonitor(P)

        rt = 3
        run(rt*second)

        print('time cost: {:.4f} s'.format(time.time()-t0))
        rE = np.sum(((s_mon.t/second)>1) & (s_mon.i<N_E))/N_E/(rt-1)
        rI = np.sum(((s_mon.t/second)>1) & (s_mon.i>=N_E))/N_I/(rt-1)
        print('No.{}/{}: rateE: {:.2f} Hz, rateI: {:.2f} Hz'.format(ii+1,n_trial,rE,rI))
        Record_rate.append([rE,rI])

    Y_all = Record_rate
    l_output = np.shape(Y_all)[1]

    if n_trial>=10:
        np.savez('Record_Neu{}_input{}d_{}output{}d_trial{}_'.format(N_E+N_I,l_para,f_Y.__name__,l_output,n_trial) \
                 +ext+'_'+time.strftime("%Y%m%d-%H%M%S"),para_all,Y_all)

    return f_Y(Y_all)
