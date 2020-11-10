#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 21:20:10 2020

@author: yaoyuzhang
"""

import sys

sys.path.append("..")

from for_figures import *
import numpy as np
import matplotlib.colors as mcolors
import matplotlib

matplotlib.use("Agg")
np.set_printoptions(suppress=True, linewidth=125)

xloc = [
    r"$S^{EE}$",
    r"$S^{EI}/S^{EE}$",
    r"$S^{IE}/S^{EE}$",
    r"$S^{II}/S^{EI}$",
    r"$\eta^{{\rm amb}}/\eta_{0}$",
    r"$\eta^{{\rm ext},E}$",
    r"$\eta^{{\rm ext},I}/\eta^{{\rm ext},E}$",
]


def pltsave(fname, opt=True, pdf=True):
    if opt == True:
        if pdf == False:
            plt.savefig(fname + ".png", dpi=300)
        elif pdf == True:
            plt.savefig(fname + ".pdf", dpi=300)


def gen_fig1a(save=False):
    dataset = np.load(
        "Dataset_NeuralData_1X_trn20000_tst10000.npy", allow_pickle=True
    ).item()
    X, Y = dataset["train"]

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    fname = "E-I_distribution_train"
    plt.figure()
    plt.hist2d(
        Y[:, 0],
        Y[:, 1],
        bins=(100, 50),
        cmap="rainbow",
        norm=mcolors.LogNorm(1, vmax=None, clip=True),
    )

    id_suit = (Y[:, 0] > 5) & (Y[:, 0] < 30)
    id_suit = id_suit & (Y[:, 1] / Y[:, 0] > 2.5) & (Y[:, 1] / Y[:, 0] < 5.5)
    print(np.sum(id_suit) / len(id_suit))

    xs = np.array([5, 30])
    ys_r = np.array([2.5, 5.5])
    plt.plot(xs, xs * ys_r[0], "-k")
    plt.plot(xs, xs * ys_r[1], "-k")
    [
        plt.plot([xs[j], xs[j]], [xs[j] * ys_r[0], xs[j] * ys_r[1]], "-k")
        for j in range(2)
    ]
    plt.xlim(0, 40)
    plt.ylim(0, 175)
    plt.xlabel(r"$r^E$ (Hz)")
    plt.ylabel(r"$r^I$ (Hz)")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("count", rotation=270)
    plt.show()
    pltsave(fname, opt=save)


def gen_fig1b(save=False, method="loglog"):
    fig_data = np.load("model_error_l1_l2_test10000_1X.npy", allow_pickle=True).item()
    globals().update(fig_data)

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    fname = "model_error"

    print(Tsamples, err_l1)

    plt.figure()
    [
        plt.__dict__[method](Tsamples, y, c)
        for y, c in zip(err_l1.transpose(), ["r-*", "b-*"])
    ]
    [
        plt.__dict__[method](Tsamples, y, c)
        for y, c in zip(err_l2.transpose(), ["r--*", "b--*"])
    ]
    #    power = 2/3
    #    plt.__dict__[method](Tsamples,3.2/Tsamples**power*Tsamples[0]**power,'-k',linewidth = 4)
    #    plt.plot(Tsamples, err_l2,'-*')
    plt.xlabel(r"$n$")
    plt.ylabel("error (Hz)")
    plt.legend(
        [
            "MAE, " + r"$r^E$",
            "MAE, " + r"$r^I$",
            "RMSE, " + r"$r^E$",
            "RMSE, " + r"$r^I$",
        ]
    )
    plt.show()
    pltsave(fname, opt=save)


def gen_fig2(save=False):
    fig_data = np.load("APT_error.npy", allow_pickle=True).item()
    globals().update(fig_data)
    fname = "APT_err"

    call_pltsettings(scale_dpi=1.5, scale=1.5, fontscale=1.6, ratio=[1, 1])
    plt.figure()
    [
        plt.scatter(pred_exp_err_all[j][:, 0], pred_exp_err_all[j][:, 1], c="k", s=10)
        for j in range(len(ratio_all))
    ]
    #    plt.scatter(tgt_all[:,0],tgt_all[:,1],c = ratio_all,cmap = 'rainbow',s = 240,marker = 'x')
    plt.scatter(tgt_all[:, 0], tgt_all[:, 1], c="r", s=240, marker="x")
    xs = np.array([5, 30])
    ys_r = np.array([2.5, 5.5])
    plt.plot(xs, xs * ys_r[0], "--k")
    plt.plot(xs, xs * ys_r[1], "--k")
    [
        plt.plot([xs[j], xs[j]], [xs[j] * ys_r[0], xs[j] * ys_r[1]], "--k")
        for j in range(2)
    ]
    #    plt.colorbar()
    #    plt.xlim([0,33])
    #    plt.ylim([0,110])
    plt.xlim(0, 40)
    plt.ylim(0, 175)
    plt.xlabel(r"$r^E$ (Hz)")
    plt.ylabel(r"$r^I$ (Hz)")
    plt.show()
    pltsave(fname, opt=save)

    call_pltsettings(scale_dpi=1.5, scale=1.5, fontscale=1.6, ratio=[1, 1])
    A = np.load("APT_results_89_tgt_25_100.npy", allow_pickle=True).item()
    # globals().update(fig_data)
    plt.figure()
    plt.scatter(A["para_cand"][:, -2], A["para_cand"][:, -1], c="c")
    plt.scatter(A["para_good"][:, -2], A["para_good"][:, -1], c="k")
    plt.xlabel(xloc[-2] + " (Hz)")
    plt.ylabel(xloc[-1])
    plt.xlim(25, 3000)
    plt.ylim(2, 6)
    pltsave(fname + "case", opt=save)


def gen_fig3(save=False):
    fig_data = np.load("derivatives.npy", allow_pickle=True).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")
    fname1 = "E_derivative_nolog"
    fname2 = "I_derivative_nolog"
    color = "hot_r"
    rs = rg2 - rg1

    id_suit = (op_pred[:, 0] > 5) & (op_pred[:, 0] < 30)
    id_suit = (
        id_suit
        & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
        & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
    )

    X = (np.arange(7) + 1)[np.newaxis, :] * np.ones([gd0.shape[0], 1])
    import matplotlib.colors as mcolors

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    # suitable
    fig, ax = plt.subplots(1, 1)
    plt.hist2d(
        X[id_suit, :].ravel(),
        (gd0 * rs)[id_suit, :].ravel(),
        bins=[0.5 + np.arange(8), 200],
        cmap=color,
        norm=mcolors.PowerNorm(0.7),
    )
    plt.plot(np.arange(8) + 0.5, np.zeros(8), "-k", linewidth=1)
    plt.ylabel(r"$\partial_{P_{i}}\hat{r}^{E}$" + " (normalized)")
    plt.ylim(-150, 150)
    #    ax.xaxis.set_major_formatter(matplotlib.ticker.IndexFormatter(xloc))
    plt.xticks(np.arange(7) + 1, xloc, rotation=-30)
    cbar = plt.colorbar()
    plt.clim(0, 30000)
    cbar.ax.get_yaxis().labelpad = 21
    cbar.ax.set_ylabel("count", rotation=270)
    plt.show()
    pltsave(fname1, opt=save)

    plt.figure()
    plt.hist2d(
        X[id_suit, :].ravel(),
        (gd1 * rs)[id_suit, :].ravel(),
        bins=[0.5 + np.arange(8), 200],
        cmap=color,
        norm=mcolors.PowerNorm(0.7),
    )
    plt.plot(np.arange(8) + 0.5, np.zeros(8), "-k", linewidth=1)
    plt.ylabel(r"$\partial_{P_{i}}\hat{r}^{I}$" + " (normalized)")
    plt.xticks(np.arange(7) + 1, xloc, rotation=-30)
    plt.ylim(-150, 150)
    cbar = plt.colorbar()
    plt.clim(0, 30000)
    cbar.ax.get_yaxis().labelpad = 21
    cbar.ax.set_ylabel("count", rotation=270)
    pltsave(fname2, opt=save)


def gen_fig4a(save=False):
    fig_data = np.load("hess_gd_all_20200115-074445.npy", allow_pickle=True).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")
    fname = "corr_plot"

    id_suit = (op_pred[:, 0] > 5) & (op_pred[:, 0] < 30)
    id_suit = (
        id_suit
        & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
        & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
    )
    #    ids = [2,0,1]
    ids = [0, 1, 2]
    cormat = np.corrcoef(np.sign(gd1[id_suit][:, [2, 3, 6]]).transpose())
    #    xyloc = [xloc[x] for x in [2,3,6]]
    xyloc = [
        r"$\frac{\partial \hat{r}^{I}}{\partial\left(S^{IE}/S^{EE}\right)}$",
        r"$\frac{\partial \hat{r}^{I}}{\partial\left(S^{II}/S^{EI}\right)}$",
        r"$\frac{\partial \hat{r}^{I}}{\partial\left(\eta^{{\rm ext},I}/\eta^{{\rm ext},E}\right)}$",
    ]
    # xyloc = [r'$\partial_{S^{IE}/S^{EE}} \hat{r}^{I}$',
    #          r'$\partial_{S^{II}/S^{EI}} \hat{r}^{I}}$',
    #         r'$\partial_{\eta^{{\rm ext},I}/\eta^{{\rm ext},E}} \hat{r}^{I}$']

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=2.4, ratio=[1, 1])
    # suitable
    fig, ax = plt.subplots(1, 1)
    #    plt.imshow(cormat-np.diag(np.diag(cormat)),cmap = 'bwr')
    plt.imshow(cormat[ids][:, ids])
    plt.clim(-1, 1)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=24)
    cbar.ax.get_yaxis().labelpad = 14
    cbar.ax.set_ylabel("correlation", rotation=270, fontsize=24)
    #    ax.xaxis.set_major_formatter(matplotlib.ticker.IndexFormatter(xloc))
    plt.xticks(np.arange(3), [xyloc[ii] for ii in ids], rotation=0)
    plt.yticks(np.arange(3), [xyloc[ii] for ii in ids], rotation=0)
    for i in range(3):
        for j in range(3):
            if cormat[i, j] <= 0:
                text = ax.text(
                    j,
                    i,
                    "{:.2f}".format(cormat[i, j]),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=24,
                )
            else:
                text = ax.text(
                    j,
                    i,
                    "{:.2f}".format(cormat[i, j]),
                    ha="center",
                    va="center",
                    color="k",
                    fontsize=24,
                )
    plt.show()
    pltsave(fname, opt=save)


def gen_fig5ab(save=False):
    fname = "importance_anal"
    fig_data = np.load("importance_analysis.npy", allow_pickle=True).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    width = 0.3
    plt.figure()
    #    plt.bar(np.arange(7)+1-width/2,pct0,width,color = 'c')
    #    plt.bar(np.arange(9)+1+width/2,pct1,width,color = 'k')
    plt.bar(np.arange(7) + 1, pct0, color="k")
    plt.xticks(np.arange(7) + 1, xloc, rotation=-30)
    plt.ylabel("ratio")
    plt.show()
    pltsave(fname + "_7", opt=save)

    plt.figure()
    #    plt.bar(np.arange(7)+1-width/2,pct0,width,color = 'c')
    #    plt.bar(np.arange(9)+1+width/2,pct1,width,color = 'k')

    plt.bar(np.arange(9) + 1, pct1, color="k")
    plt.xticks(np.arange(9) + 1, xloc + [r"$\hat{r}^E$", r"$\hat{r}^I$"], rotation=-30)
    plt.ylabel("ratio")
    plt.show()
    pltsave(fname + "_9", opt=save)


def gen_fig5c(save=False):
    matplotlib.use("Agg")
    #    fig_data = np.load('hess_gd_all_20200115-074445.npy',allow_pickle=True).item()
    fig_data = np.load(
        "hess_gd_all_20200130-121345_fullrange.npy", allow_pickle=True
    ).item()
    globals().update(fig_data)

    id_suit = (op_pred[:, 0] > 5) & (op_pred[:, 0] < 30)
    id_suit = (
        id_suit
        & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
        & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
    )

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    fname = "derivative_analysis_add2"
    plt.figure()
    plt.scatter(
        para_cand[id_suit][:, 2],
        para_cand[id_suit][:, 6],
        c=gd1[id_suit][:, 6],
        cmap="bwr",
        s=1,
    )
    #    plt.scatter(para_cand[id_suit][:,2]*para_cand[id_suit][:,1]*para_cand[id_suit][:,0]**2,para_cand[id_suit][:,6],c=gd1[id_suit][:,6],cmap='bwr',s = 1)
    plt.ylabel(r"$S^{IE}/S^{EE}$")
    plt.xlabel(r"$\eta^{ext,I}/\eta^{ext,E}$")
    cbar = plt.colorbar()
    plt.clim(-3, 3)
    cbar.ax.get_yaxis().labelpad = 18
    cbar.ax.set_ylabel("derivative", rotation=270)
    pltsave(fname + "_analog", opt=save)


def gen_fig5d(save=False):
    matplotlib.use("Agg")

    #    fig_data = np.load('hess_gd_all_20200115-074445.npy',allow_pickle=True).item()
    fig_data = np.load(
        "hess_gd_all_20200130-121345_fullrange.npy", allow_pickle=True
    ).item()
    globals().update(fig_data)

    id_suit = (op_pred[:, 0] > 1) & (op_pred[:, 0] < 30)
    id_suit = (
        id_suit
        & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
        & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
    )
    print(np.sum(id_suit) / len(id_suit))

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    fname = "derivative_analysis"
    plt.figure()
    plt.scatter(
        para_cand[id_suit][:, 5],
        op_pred[id_suit][:, 0],
        c=gd1[id_suit][:, 6],
        cmap="bwr",
        s=1,
    )
    plt.ylim(0, 30)
    plt.ylabel(r"$\hat{r}^E$ (Hz)")
    plt.xlabel(r"$\eta^{ext,E}$ (Hz)")

    cbar = plt.colorbar()
    plt.clim(-3, 3)
    cbar.ax.get_yaxis().labelpad = 18
    cbar.ax.set_ylabel("derivative", rotation=270)

    pltsave(fname + "_analog", opt=save)


def gen_fig6(save=False):
    #    fig_data = np.load('hess_gd_all_20200115-074445.npy',allow_pickle=True).item()
    fig_data = np.load(
        "hess_gd_all_20200130-121345_fullrange.npy", allow_pickle=True
    ).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")
    fname = "hess_plot"
    rs = rg2 - rg1
    X = (np.arange(7) + 1)[np.newaxis, :] * np.ones([gd0.shape[0], 1])

    id_suit = (op_pred[:, 0] > 5) & (op_pred[:, 0] < 30)
    id_suit = (
        id_suit
        & (op_pred[:, 1] / op_pred[:, 0] > 2.5)
        & (op_pred[:, 1] / op_pred[:, 0] < 5.5)
    )

    Yp1_suit = (0.5 * hess0_diag_all * (rs ** 2))[id_suit, :]
    Yp2_suit = (0.5 * hess1_diag_all * (rs ** 2))[id_suit, :]
    # color = 'Reds'
    color = "hot_r"

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    plt.figure()
    plt.plot(np.arange(8) + 0.5, np.zeros(8), "-k", linewidth=1)
    plt.hist2d(
        X[id_suit, :].ravel(),
        Yp1_suit.ravel(),
        bins=[0.5 + np.arange(8), 800],
        cmap=color,
        norm=mcolors.PowerNorm(0.7),
    )
    plt.ylabel(r"$\partial_{P_{i}}^{2}\hat{r}^{E}$" + " (normalized)")
    plt.xticks(np.arange(7) + 1, xloc, rotation=-30)
    plt.show()
    plt.ylim(-50, 150)
    cbar = plt.colorbar()
    plt.clim(0, 25000)
    cbar.ax.get_yaxis().labelpad = 21
    cbar.ax.set_ylabel("count", rotation=270)
    pltsave(fname + "_E_nolog", opt=save)

    plt.figure()
    plt.plot(np.arange(8) + 0.5, np.zeros(8), "-k", linewidth=1)
    plt.hist2d(
        X[id_suit, :].ravel(),
        Yp2_suit.ravel(),
        bins=[0.5 + np.arange(8), 500],
        cmap=color,
        norm=mcolors.PowerNorm(0.7),
    )
    plt.ylabel(r"$\partial_{P_{i}}^{2}\hat{r}^{I}$" + " (normalized)")
    plt.xticks(np.arange(7) + 1, xloc, rotation=-30)
    plt.show()
    plt.ylim(-50, 150)
    cbar = plt.colorbar()
    plt.clim(0, 25000)
    cbar.ax.get_yaxis().labelpad = 21
    cbar.ax.set_ylabel("count", rotation=270)
    pltsave(fname + "_I_nolog", opt=save)


def gen_fig7a(save=False):
    fname = "couple_hess_dist"
    fig_data = np.load("couple_hess_dist.npy", allow_pickle=True).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")

    tgs = np.asarray([[-11, 0], [-3, 10], [4, 30]])
    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.9, ratio=[1, 1])
    plt.figure()
    plt.scatter(mean1, mean2, s=5)
    plt.ylim(-30, 100)
    plt.xlim(-25, 15)
    plt.plot([0, 50], [0, 0], "-k")
    plt.plot([0, 0], [0, -50], "-k")
    plt.scatter(tgs[:, 0], tgs[:, 1], c="k", marker="x", s=150)
    plt.xlabel(r"$m_1(P^-)$" + " (Hz)")
    plt.ylabel(r"$m_2(P^-)$" + " (Hz)")
    plt.show()
    pltsave(fname, opt=save)

    # #    tg = [-10,1]
    #     tg = [2,20]
    #     tg = [-20,0]
    #     tg = [-10,40]
    #     tg = [0,60]
    #     tg = [10,80]
    #     tg = [0,0]
    tg = [4, 0]
    # #    tgs = [[-12,0],[-4,12],[4,30]]
    #     tg = tgs[2]
    idx = np.argmin((mean1 - tg[0]) ** 2 + (mean2 - tg[1]) ** 2)
    print(mean1[idx], mean2[idx])
    para_idx = para_real[idx]
    para_idx


def gen_fig7bcd(save=False):
    fname = "gain_curves"
    fig_data = np.load("gain_curves_data_20200218-151246.npy", allow_pickle=True).item()
    globals().update(fig_data)
    print(fig_data.keys(), "are released into workspace!")

    nm = 30
    for i in range(len(params)):
        para_test = params[i]
        para_test = para_test * np.ones([nm, 1])
        para_test[:, 5] = para_test[:, 5] * np.linspace(0, 1, nm)

        call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.9, ratio=[0.6, 1])
        plt.figure()
        plt.plot(para_test[:, 5], np.asarray(op_all[i])[:, 0], "--b", lw=2)
        plt.plot(para_test[:, 5], np.asarray(rate_all[i])[:, 0], "-k", lw=2)
        plt.ylim(0, 30)
        plt.xlabel(r"$\eta^{{\rm ext},E}$ (Hz)")
        plt.ylabel(r"$r^{E}$ (Hz)")
        # plt.show()
        pltsave(fname + "_{}_ylim".format(i + 1), opt=save)


def gen_fig8(save=False, method="loglog"):
    fig_data = np.load(
        "model_error_l1_l2_test10000_10X_1X.npy", allow_pickle=True
    ).item()
    globals().update(fig_data)

    call_pltsettings(scale_dpi=1, scale=1.5, fontscale=1.6, ratio=[1, 1])
    fname = "model_error_10X_1X"

    print(Tsamples, err_l1)

    plt.figure()
    [
        plt.__dict__[method](Tsamples, y, c)
        for y, c in zip(err_l1.transpose(), ["r-*", "b-*"])
    ]
    [
        plt.__dict__[method](Tsamples, y, c)
        for y, c in zip(err_l2.transpose(), ["r--*", "b--*"])
    ]
    #    power = 2/3
    #    plt.__dict__[method](Tsamples,3.2/Tsamples**power*Tsamples[0]**power,'-k',linewidth = 4)
    #    plt.plot(Tsamples, err_l2,'-*')
    plt.xlabel(r"$n$")
    plt.ylabel("error (Hz)")
    plt.legend(
        [
            "MAE, " + r"$r^E$",
            "MAE, " + r"$r^I$",
            "RMSE, " + r"$r^E$",
            "RMSE, " + r"$r^I$",
        ]
    )
    plt.show()
    pltsave(fname, opt=save)


# gen_fig1a(save=True)
# gen_fig1b(save=True)
# gen_fig2(save=True)
# gen_fig3(save=True)
# gen_fig4a(save=True)
# gen_fig5ab(save=True)
# gen_fig5c(save=True)
# gen_fig5d(save=True)
# gen_fig6(save=True)
# gen_fig7a(save=True)
# gen_fig7bcd(save=True)
# gen_fig8(save=True)
