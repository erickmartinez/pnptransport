# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:24:24 2019

@author: Erick
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.animation as manimation
from matplotlib.ticker import ScalarFormatter
from scipy import integrate
import pnptransport.utils as utils
import h5py
import os
import pandas as pd
import platform
import matplotlib.gridspec as gridspec
import re

dataPath = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200831\results'
dataFile = r'constant_source_flux_96_85C_1E+10pcm2_z1E-05ps_DSF1E-14_1E+04Vcm_h1E-12_m1E+00_rt12h_rv-1E+04Vcm.h5'
results_path = 'videos'

max_time = 96  # hr

plot_vfb = False

plotStyle = {
    'font.size': 12,
    'font.family': 'Arial',
    'font.weight': 'regular',
    'legend.fontsize': 12,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',  # 'Arial:italic',
    'mathtext.cal': 'Times New Roman:italic',  # 'Arial:italic',
    'mathtext.bf': 'Times New Roman:bold',  # 'Arial:bold',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4.0,
    'xtick.major.width': 1.5,
    'ytick.major.size': 4.0,
    'ytick.major.width': 1.5,
    'xtick.minor.size': 2.5,
    'xtick.minor.width': 0.7,
    'ytick.minor.size': 2.5,
    'ytick.minor.width': 0.7,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'lines.markeredgewidth': 1.0,
    'axes.labelpad': 5.0,
    'axes.labelsize': 14,
    'axes.labelweight': 'regular',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.titlepad': 8,
    'figure.titleweight': 'bold',
    'figure.dpi': 100
}

q_red = 1.6021766208  # x 1E-19 C
e0_red = 8.854187817620389  # x 1E-12 C^2 / J m


def update_line(n, line_, full_file, x_1, x_2, time_s, tau, v_fb):
    i_max = (np.abs(time - max_time * 3600.)).argmin()
    with h5py.File(full_file, 'r') as hf_:
        grp_sinx_ = hf_['/L1']
        grp_si_ = hf_['/L2']
        ct_ds = 'ct_{0:d}'.format(n)
        c_1 = np.array(grp_sinx_['concentration'][ct_ds])
        c_2 = np.array(grp_si_['concentration'][ct_ds])
        tn = time_s[n]

        line_[0].set_data(x_1, c_1)
        line_[1].set_data(x_2, c_2)
        if n > 0:
            line_[2].set_data(np.array(time_s[0:n] / 3600), np.array(v_fb[0:n]))
        line_[3].set_text(r'{0}'.format(utils.format_time_str(tn)))

    # if tn >= tau:
    #     line_[0].set_color('r')
    #     line_[1].set_color('r')

    #    Cint = integrate.simps(x1/1000*c1,x1/1000)*1E-8
    #    vfbi = -q_red*Cint/(7*e0_red)*1E-5
    #    vfb[i] = vfbi
    #    print('Updating time step {0}/{1}, vfb = {2:.3f}'.format(i,len(time_s),vfbi))
    print('Updating time step {0}/{1}'.format(n, i_max))
    return line_


if __name__ == '__main__':

    output_path = os.path.join(dataPath, results_path)

    file_tag = os.path.splitext(dataFile)[0]
    fullfile = os.path.join(dataPath, dataFile)
    if platform.system() == 'Windows':
        fullfile = r'\\?\\' + fullfile
        output_path = r'\\?\\' + output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # x1 = None
    # x2 = None
    # c1 = None
    # c2 = None

    finite_source = False

    with h5py.File(fullfile, 'r') as hf:
        grp_time = hf['time']
        time = np.array(hf['time'])
        vfb = np.array(hf['vfb'])
        min_vfb = np.amin(vfb)
        max_vfb = np.amax(vfb)
        h = hf['time'].attrs['h']

        TempC = grp_time.attrs['temp_c']
        try:
            Cs = grp_time.attrs['Csource']
        except Exception as e:
            finite_source = True
            surface_concentration = float(grp_time.attrs['surface_concentration'])
            h_surface = float(grp_time.attrs['h_surface'])
        Cbulk = grp_time.attrs['Cbulk']
        #
        # h = grp_time.attrs['h']
        # m = grp_time.attrs['m']

        grp_sinx = hf['/L1']
        grp_si = hf['/L2']

        er = grp_sinx.attrs['er']
        E1 = grp_sinx.attrs['electric_field_eff'] * er
        D1 = grp_sinx.attrs['D']
        V1 = grp_sinx.attrs['stress_voltage']

        D2 = grp_si.attrs['D']

        x1 = np.array(grp_sinx['x']) * 1000
        x2 = np.array(grp_si['x'])

        L1 = np.amax(x1)
        x1 = x1 - L1
        x2 = x2 - L1 / 1000

        tau_c = utils.tau_c(D1, E1 / er, L1 * 1E-7, TempC)

        c1 = np.array(grp_sinx['concentration']['ct_0'])
        c2 = np.array(grp_si['concentration']['ct_0'])

        # Plot style parameters
    mpl.rcParams.update(plotStyle)
    imax = (np.abs(time - max_time * 3600)).argmin()

    fig1 = plt.figure()
    fig1.set_size_inches(6.5, 5.5, forward=True)
    fig1.subplots_adjust(hspace=0.35, wspace=0.35)
    gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig1, hspace=0.6, wspace=0.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=[0.5, 1.0], subplot_spec=gs0[0], wspace=0)
    gs01 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1])
    gs0.tight_layout(fig1, rect=[0.5, 1, 1., 1.])

    ax1 = fig1.add_subplot(gs00[0, 0])
    ax2 = fig1.add_subplot(gs00[0, 1])
    ax3 = fig1.add_subplot(gs01[0, 0])

    ax1.set_facecolor((0.89, 0.75, 1.0))
    ax2.set_facecolor((0.82, 0.83, 1.0))

    ph1, = ax1.plot(x1, c1, color='C0')
    ph2, = ax2.plot(x2, c2, color='C0')
    ph3, = ax3.plot(np.array(time[0] / 3600.), np.array(vfb[0]), color='C1')

    ax1.set_yscale('log')
    ax2.set_yscale('log')

    ax1.set_xlim([np.amin(x1), np.amax(x1)])
    ax2.set_xlim([np.amin(x2), 1.0])
    ax1.set_ylim([1E12, 1E22])
    ax2.set_ylim([1E12, 1E22])

    ax3.set_xlim(0, time[imax] / 3600.)
    ax3.set_ylim(bottom=min_vfb * 1.2, top=max_vfb * 1.2)

    ax1.tick_params(labelbottom=False, bottom=True, top=False, right=False, which='left')
    ax2.tick_params(labelbottom=True, labelleft=False, left=False, bottom=True, top=False, right=True, which='right')
    ax2.yaxis.set_ticks_position('right')
    ax3.tick_params(labelleft=True, left=True)

    ax1.set_ylabel("[Na$^{+}$] (cm$^{-3}$)")
    ax1.set_xlabel("Depth (nm)")

    ax2.set_xlabel(r'Depth ($\mathregular{\mu}$m)')

    ax3.set_xlabel("Time (h)")
    ax3.set_ylabel(r'$\Delta V_{\mathrm{FB}}$ (V)')
    # ax3.axvline(x=tau_c / 3600, ls='--', dashes=(3, 2), color='k', lw=1.5)
    # ax3.text(
    #     tau_c / 3600, np.amax(vfb), '\n $\\tau_c = {0:.1f}$ h'.format(tau_c / 3600),
    #     horizontalalignment='left',
    #     verticalalignment='top',
    #     # transform=ax3.transAxes,
    #     fontsize=14,
    #     color='r',
    #     zorder=3
    # )

    time_txt = ax1.text(
        0.05, 0.05, utils.format_time_str(0.0),
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax1.transAxes
    )

    if not finite_source:
        params1_str = '$C_s$ = %s cm$^{-3}$\nD = %s cm$^2$/s' % (
            utils.latex_format(Cs, digits=1), utils.latex_format(D1, digits=1)
        )
    else:
        params1_str = r'$S = %s \; \mathregular{cm^{-2}}$' % (
            utils.latex_order_of_magnitude(surface_concentration, dollar=False)
        )
        params1_str += '\n'
        params1_str += r'$D = $ %s $\mathregular{cm^2/s}$' % (
            utils.latex_format(D1, digits=1)
        )

    params1_txt = ax1.text(0.05, 0.95, params1_str,
                           horizontalalignment='left',
                           verticalalignment='top',
                           transform=ax1.transAxes,
                           fontsize=11,
                           color='b')

    h_exp = int(("%e" % h).split('e')[1])
    params2_str = 'D = %s cm$^2$/s\n$h$ = 10$^{%d}$ cm/s' % (utils.latex_format(D2, digits=1), h_exp)
    params2_txt = ax2.text(0.05, 0.95, params2_str,
                           horizontalalignment='left',
                           verticalalignment='top',
                           transform=ax2.transAxes,
                           fontsize=11,
                           color='b')

    ax1.set_title(r'SiN$_{\mathregular{x}}$')  #, %s$ MV/cm' % (utils.latex_format(E1, digits=1)))
    # ax1.set_title(r'SiN$_{\mathregular{x}}$, 0.5 MV/cm' )
    ax2.set_title('Si')

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-2, 2))
    locmaj1 = mpl.ticker.LogLocator(base=10.0, numticks=5)
    locmin1 = mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1)

    # ax1.yaxis.set_ticks_position('both')

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(5, prune='upper'))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1.yaxis.set_major_locator(locmaj1)
    ax1.yaxis.set_minor_locator(locmin1)

    # Axis 2
    locmaj2 = mpl.ticker.LogLocator(base=10.0, numticks=5)
    locmin2 = mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1)

    ax2.xaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax2.yaxis.set_major_locator(locmaj2)
    ax2.yaxis.set_minor_locator(locmin2)
    #    ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax3.xaxis.set_major_locator(mticker.MaxNLocator(12, prune=None))
    ax3.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax3.yaxis.set_major_formatter(xfmt)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    ax3.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    plt.tight_layout()
    # plt.show()

    line = [ph1, ph2, ph3, time_txt]

    # FFMpegWriter = manimation.writers['ffmpeg']
    plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.4-Q16\magick.exe'
    metadata = dict(title='Simulated SIMS profile', artist='Matplotlib', comment='Time dependent profile')
    # writer = FFMpegWriter(fps=12, metadata=metadata, extra_args=['-vcodec', 'libx264'])
    ani = manimation.FuncAnimation(
        fig1, update_line, blit=True, interval=1,
        repeat=False, frames=np.arange(0, imax, 1),
        fargs=(line, fullfile, x1, x2, time, tau_c, vfb)
    )

    # ft = os.path.join(output_path, file_tag + '.mp4')
    ft = os.path.join(output_path, file_tag + '.gif')

    # ani.save(ft, writer=writer, dpi=300)
    ani.save(ft, writer='imagemagick', fps=10)

