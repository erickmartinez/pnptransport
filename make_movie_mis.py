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
import matplotlib.gridspec as gridspec
from pnptransport import utils
import h5py
import os
import pandas as pd
import platform

data_path = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\figures\zero-flux'
results_sub_folder = 'videos'
data_file = r'single_layer_D1E-16cm2ps_Cs1E+16cm3_T80_time24hr_pnp.h5'

plot_vfb = True

plot_style = {
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
    'axes.labelpad': 6.0,
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


def update_line(n, _line, _hf_file_name, x_sin, time_s, tau, v_fb):
    with h5py.File(_hf_file_name, 'r') as _hf:
        _grp_sinx = _hf['/L1']
        ct_ds = 'ct_{0:d}'.format(n)
        c_sin = np.array(_grp_sinx['concentration'][ct_ds])
    _time_i = time_s[n]
    _line[0].set_data(x_sin, c_sin)
    if n > 0:
        _line[1].set_data(np.array(time_s[0:n] / 3600), np.array(v_fb[0:n]))
    _line[2].set_text(r'{0}, $\tau_c$ = {1:.1f} h'.format(utils.format_time_str(_time_i), tau / 3600))


    if _time_i >= tau:
        _line[0].set_color('r')
    print('Updating time step {0}/{1}'.format(n, len(time_s)))
    return _line


if __name__ == '__main__':
    filetag = os.path.splitext(data_file)[0]
    h5_file = os.path.join(data_path, data_file)
    if platform.system() == 'Windows':
        h5_file = r'\\?\\' + h5_file

    results_path = os.path.join(os.path.dirname(h5_file), results_sub_folder)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with h5py.File(h5_file, 'r') as hf:
        grp_time = hf['time']
        time = np.array(hf['time'])
        vfb = np.array(hf['vfb'])
        TempC = grp_time.attrs['temp_c']
        # Cs = grp_time.attrs['Csource']
        # Cbulk = grp_time.attrs['Cbulk']
        grp_sinx = hf['L1']
        er = grp_sinx.attrs['er']
        E1 = grp_sinx.attrs['electric_field_eff'] * er
        D1 = grp_sinx.attrs['D']
        V1 = grp_sinx.attrs['stress_voltage']
        x1 = np.array(grp_sinx['x']) * 1000
        L1 = np.amax(x1)
        c1 = np.array(grp_sinx['concentration']['ct_0'])
    Cs = c1[0]
    tau_c = utils.tau_c(D1, E1 / er, L1 * 1E-7, TempC)

    # Plot style parameters
    mpl.rcParams.update(plot_style)

    fig = plt.figure()
    fig.set_size_inches(6.0, 5.5, forward=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.35)
    gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.3)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, width_ratios=[1.0], subplot_spec=gs0[0], wspace=0)
    gs01 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[1], )
    gs2 = fig.add_gridspec(nrows=1, ncols=1)
    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig.add_subplot(gs01[0, 0])

    #    ax1.set_facecolor((0.89, 0.75, 1.0))

    ph1, = ax1.plot(x1, c1, color='C0')
    ph2, = ax2.plot(np.array(time[0] / 3600), np.array(vfb[0]), color='C1')

    ax1.set_xlim([0, L1])
    ax1.set_yscale('log')
    ax1.set_ylim([1E12, 1E22])

    ax2.set_xlim(0, time[-1] / 3600)

    min_vfb = np.amin(vfb)
    max_vfb = np.amax(vfb)
    ax2.set_ylim(bottom=min_vfb * 1.2, top=max_vfb * 1.2)

    ax1.tick_params(labelbottom=False, bottom=True, top=False, right=False, which='left')

    ax1.set_xlabel("Depth (nm)")
    ax1.set_ylabel("[Na$^{+}$] (cm$^{-3}$)")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("$\Delta V_{\mathrm{fb}}$ (V)")

    time_txt = ax1.text(0.05, 0.05, utils.format_time_str(0.0),
                        horizontalalignment='left',
                        verticalalignment='bottom',
                        transform=ax1.transAxes)

    params_str = 'C$_s$ = %s cm$^{-3}$\nD = %s cm$^2$/s' % (
        utils.latex_format(Cs, digits=1), utils.latex_format(D1, digits=1)
    )
    params_txt = ax1.text(
        0.05, 0.95, params_str,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax1.transAxes,
        fontsize=12,
        color='b'
    )

    ax1.set_title('SiN$_{\mathregular{x}}$, E = %s MV/cm' % (utils.latex_format(E1, digits=1)))

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-4, 4))
    locmaj = mpl.ticker.LogLocator(base=10.0, numticks=5)
    locmin = mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1)

    ax1.yaxis.set_ticks_position('both')

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(7, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)
    #    ax1.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    plt.tight_layout()


    line = [ph1, ph2, time_txt]

    FFMpegWriter = manimation.writers['ffmpeg']
    plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.4-Q16\magick.exe'
    metadata = dict(title='Simulated SIMS profile', artist='Matplotlib',
                    comment='Time dependent profile')
    # writer = FFMpegWriter(fps=12, metadata=metadata, extra_args=['-vcodec', 'libx264'])
    ani = manimation.FuncAnimation(
        fig, update_line, blit=True, interval=5,
        repeat=False, frames=np.arange(0, len(time), 5),
        fargs=(line, h5_file, x1, time, tau_c, vfb)
    )

    # plt.show()

    ft = os.path.join(results_path, filetag + '.gif')
    # ani.save(ft, writer=writer, dpi=300)
    ani.save(ft, writer='imagemagick', fps=10)



