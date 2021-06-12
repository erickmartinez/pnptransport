import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec
import itertools
from pnptransport import utils
import h5py
import os

file_index_csv = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\figures\vfb_for_different_h\vfb_h.csv'
experimental_file = r'\\?\G:\Shared drives\FenningLab2\Projects\PVRD1\DataProcessing\data_fitting\CVfitting\python\vfb_fit\erf\results\fitting_results_sinx\MU2_AC1_D205D206D207D208_ANa1_D205D206D207D208_smoothed_cont_minus_clean_6\MU2_AC1_D205D206D207D208_ANa1_D205D206D207D208_smoothed_cont_minus_clean_6_erf_fit.h5'

if __name__ == '__main__':
    with open('plotstyle.json', 'r') as f:
        mpl.rcParams.update(json.load(f)['defaultPlotStyle'])
    input_files_df = pd.read_csv(file_index_csv)
    cm = cmap.get_cmap('cool_r')

    with h5py.File(experimental_file, 'r') as hf:
        vfb_exp_ds = hf['vfb_data']
        time_exp = vfb_exp_ds['time_s'] / 3600
        vfb_exp = vfb_exp_ds['vsh']
        vfb_std = vfb_exp_ds['vsh_std']

    nplots = len(input_files_df)  # + len(literature_files_df)
    normalize = mpl.colors.Normalize(vmin=0, vmax=(nplots - 1))
    plot_colors = [cm(normalize(i)) for i in range(nplots)]
    plot_marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p', '*'))

    fig = plt.figure()
    fig.set_size_inches(4.5, 4.5, forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.4, height_ratios=[0.5, 0.5])
    gs00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs0[0], wspace=0.0, hspace=0.0, width_ratios=[0.3, 0.7]
    )
    gs10 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs0[1], wspace=0.0, hspace=0.0,
    )

    ax_c_0 = fig.add_subplot(gs00[0, 0])
    ax_c_1 = fig.add_subplot(gs00[0, 1])
    ax_v_0 = fig.add_subplot(gs10[0, 0])

    tau_c = np.empty(len(input_files_df))

    for i, r in input_files_df.iterrows():
        h5_file = r['h5_file']
        with h5py.File(h5_file, 'r') as hf:
            time_s = np.array(hf.get('time'))
            time_h = time_s / 3600.
            idx_96 = (np.abs(time_h - 96)).argmin()
            x1 = np.array(hf['/L1/x'])
            L1 = np.amax(x1)
            D1 = hf['L1'].attrs['D']
            temp = hf['/time'].attrs['temp_c']
            EField = hf['/L1'].attrs['electric_field_eff']
            bias = hf['/L1'].attrs['stress_voltage']
            tau_c[i] = utils.tau_c(D=D1, L=L1 * 1E-4, E=EField, T=temp)
            c_ds = 'ct_{0}'.format(idx_96)
            x_sin = np.array(hf['L1/x']) * 1000.
            thickness_sin = np.max(x_sin)
            c_1 = np.array(hf['L1/concentration/{0}'.format(c_ds)])
            x_si = np.array(hf['L2/x']) - thickness_sin / 1000.
            x_sin = x_sin - thickness_sin
            thickness_si = np.amax(x_si)
            c_2 = np.array(hf['L2/concentration/{0}'.format(c_ds)])
            vfb = np.array(hf['vfb'])
            ax_c_0.plot(x_sin, c_1, color=plot_colors[i])
            ax_c_1.plot(x_si, c_2, color=plot_colors[i])
            lbl = '${0}$'.format(utils.latex_order_of_magnitude(r['h']))
            ax_v_0.plot(time_h, -vfb / vfb.min(), color=plot_colors[i], label=lbl)

    idxs = time_exp <= tau_c.mean() * 10
    t_exp = time_exp[idxs]
    v_exp = vfb_exp[idxs]

    ax_v_0.plot(t_exp, -v_exp / v_exp.min(),
                marker='o',
                markerfacecolor='none',
                fillstyle='none',
                markeredgecolor='tab:green',
                ms=7,
                lw=2.25,
                zorder=1,
                ls='none',
                color='tab:green',
                label='Trac-BTS')

    leg = ax_v_0.legend(loc='best', frameon=True)
    ax_v_0.set_xlim(left=0, right=96)
    ax_c_0.set_xlim(left=np.amin(x_sin), right=np.amax(x_sin))
    ax_c_1.set_xlim(left=np.amin(x_si), right=np.amax(x_si))
    # Axis labels
    ax_c_0.set_xlabel(r'Depth (nm)')
    ax_c_0.set_ylabel(r'[Na] ($\mathregular{cm^{-3}}$)')
    # Title to the sinx axis
    # ax_c_0.set_title(r'$10^4\; \mathregular{{V/cm}}, 85\; \mathrm{{°C}}$', fontweight='regular')
    # Set the ticks for the Si concentration profile axis to the right
    ax_c_1.yaxis.set_ticks_position('right')
    # Title to the si axis
    # ax_c_1.set_title(r'$D_{{\mathrm{{SF}}}} = 10^{-14}\; \mathregular{{cm^2/s}},\; E=0$', fontweight='regular')
    ax_c_1.set_xlabel(r'Depth (um)')
    # Log plot in the y axis
    ax_c_0.set_yscale('log')
    ax_c_1.set_yscale('log')
    ax_c_1.yaxis.tick_right()
    ax_c_1.tick_params(axis='y', left=False, labelright=True, right=True)
    ax_c_0.set_ylim(bottom=1E12, top=1E17)
    ax_c_1.set_ylim(bottom=1E12, top=1E17)
    # Set the ticks for the SiNx log axis
    ax_c_0.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
    ax_c_0.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1))
    # Set the ticks for the Si log axis
    ax_c_1.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
    ax_c_1.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1))
    # ax_c_1.tick_params(axis='y', left=False, labelright=False)
    # Configure the ticks for the x axis
    ax_c_0.xaxis.set_major_locator(mticker.MaxNLocator(3, prune=None))
    ax_c_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax_c_1.xaxis.set_major_locator(mticker.MaxNLocator(6, prune='lower'))
    ax_c_1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax_v_0.set_xlabel(r'Time (h)')
    ax_v_0.set_ylabel(r'$\Delta V_{\mathrm{FB}}/ \Delta V_{\mathrm{min}}$ ')

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))
    ax_v_0.yaxis.set_major_formatter(xfmt)

    fig.tight_layout()
    save_dir = os.path.dirname(file_index_csv)
    plt.savefig(os.path.join(save_dir, 'vfb_at_h.svg'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, 'vfb_at_h.png'), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
