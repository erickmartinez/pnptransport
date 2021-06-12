# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 06:21:36 2020

@author: Erick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import platform
import os
import matplotlib.gridspec as gridspec
import json

results_folder = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121'

data_files = [
    {
        'filepath': r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121\ofat_parameter_dsf.csv',
        'sweep_variable': '$D_{\mathrm{SF}}$',
        'sweep_variable_units': 'cm$\mathrm{^{2}}$/s',
        'unit_prefix': None,
        'columns': ['D (cm^2/s)', 't 1000 (s)'],
        'sweep_values': [1.0E-16, 1.0E-15, 1.0E-14],
    },
    {
        'filepath': r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121\ofat_parameter_e.csv',
        'sweep_variable': '$E$',
        'sweep_variable_units': 'V/cm',
        'unit_prefix': 'M',
        'columns': ['$E$ (V/cm)', 't 1000 (s)'],
        'sweep_values': [1E3, 1E4, 1E5],
    },
    {
        'filepath': r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121\ofat_parameter_sigma_s.csv',
        'sweep_variable': '$S_0$',
        'sweep_variable_units': 'cm$^{-2}$',
        'columns': ['$S$ (cm$\mathregular{^{-2}}$)', 't 1000 (s)'],
        'sweep_values': [1e+10, 1e+11, 1e+12],
    },
    {
        'filepath': r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121\ofat_parameter_zeta.csv',
        'sweep_variable': '$k$',
        'sweep_variable_units': '1/s',
        'columns': ['$k$ (s$\mathregular{^{-1}}$)', 't 1000 (s)'],
        'sweep_values': [1e-7, 1e-5, 1e-4],
    },
    {
        'filepath': r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_comparison_20201121\ofat_parameter_h.csv',
        'sweep_variable': '$h$',
        'sweep_variable_units': 'cm/s',
        'unit_prefix': None,
        'columns': ['$h$ (cm/s)', 't 1000 (s)'],
        'sweep_values': [1E-12, 1E-10, 1E-8],
    }

]

bar_width = 0.3


def autolabel(rects, data_files_, column_index, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    # print(column_index)

    for i, rect in enumerate(rects):
        height = rect.get_height()
        fi = data_files_[i]

        sv = fi['sweep_values'][column_index]
        val_str = '{0:.1E}'.format(sv)
        val_arr = val_str.split('E')
        val_arr = np.array(val_arr, dtype=float)
        lbl = r' ${{ 10^{{{0:.0f}}} }}$'.format(
            val_arr[1]
        )
        # if fi['sweep_variable'] == '$E$':
        #     sv = sv / 1E6
        #     if sv * 10 > 0.1:
        #         lbl = ' {0:.1f} MV/cm'.format(sv)
        #     elif sv <= 0.01:
        #         lbl = ' {0:.0f} kV/cm'.format(sv * 1000)
        if height < 40:
            ax1.text(
                rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height, lbl,
                ha=ha[xpos], va='bottom', rotation=90, fontsize=11, color=lighten_color(bar_colors[column_index], 1.5),
                fontweight=800
            )
        else:
            ax1.text(
                rect.get_x() + rect.get_width() * offset[xpos], 2, lbl,
                ha=ha[xpos], va='bottom', rotation=90, fontsize=11, color=lighten_color(bar_colors[column_index], 1.5),
                fontweight=800
            )

def autolabel_rsh(rects, data_files_, column_index, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
    # print(column_index)

    for i, rect in enumerate(rects):
        height = rect.get_height()
        fi = data_files_[i]

        sv = fi['sweep_values'][column_index]
        val_str = '{0:.1E}'.format(sv)
        val_arr = val_str.split('E')
        val_arr = np.array(val_arr, dtype=float)
        lbl = r' ${{ 10^{{{0:.0f}}} }}$'.format(
            val_arr[1]
        )
        if height < 200:
            ax2.text(
                rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height, lbl,
                ha=ha[xpos], va='bottom', rotation=90, fontsize=11, color=lighten_color(bar_colors[column_index], 1.5),
                fontweight=800
            )
        else:
            ax2.text(
                rect.get_x() + rect.get_width() * offset[xpos], 10, lbl,
                ha=ha[xpos], va='bottom', rotation=90, fontsize=11, color=lighten_color(bar_colors[column_index], 1.5),
                fontweight=800
            )


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__ == '__main__':

    c_map1 = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.Normalize(vmin=0, vmax=2)
    bar_colors = [c_map1(normalize(i)) for i in range(3)]
    with open('plotstyle.json', 'r') as style_file:
        mpl.rcParams.update(json.load(style_file)['defaultPlotStyle'])
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    if platform.system() == 'Windows':
        results_folder = r'\\?\\' + results_folder

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    fig = plt.figure(1)
    fig.set_size_inches(6.5, 3.5, forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, height_ratios=[1],
        subplot_spec=gs0[0]
    )

    fig_mr = plt.figure(2)
    fig_mr.set_size_inches(6.5, 3.5, forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    gs0_mr = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_mr, width_ratios=[1])
    gs00_mr = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, height_ratios=[1],
        subplot_spec=gs0_mr[0]
    )

    n_variables = len(data_files)
    sweep_values = np.empty(
        n_variables, dtype=np.dtype([('low', 'd'), ('medium', 'd'), ('high', 'd')])
    )
    
    max_rsh = np.empty(
        n_variables, dtype=np.dtype([('low', 'd'), ('medium', 'd'), ('high', 'd')])
    )
    
    xticks = list(range(n_variables))
    xtick_labels = []
    ax1 = fig.add_subplot(gs00[0, 0])
    ax2 = fig_mr.add_subplot(gs00_mr[0, 0])
    # ax1b = fig.add_subplot(gs00[0, 0], sharex=ax1)
    ax2.set_yscale('log')

    ax1.set_ylim(top=60.)
    # ax1b.set_ylim(bottom=30, top=200)

    zorder = 0
    for i, fi in enumerate(data_files):
        fn = fi['filepath']
        if platform.system() == 'Windows':
            fn = r'\\?\\' + fn
        xtick_labels.append(
            '{0} ({1})'.format(fi['sweep_variable'], fi['sweep_variable_units'])
        )
        df = pd.read_csv(filepath_or_buffer=fn, usecols=[0, 1, 2])
        df_cols = df.columns
        param_values = df.iloc[:, 0]
        sweep_values_str = ['{0:.1E}'.format(v) for v in fi['sweep_values']]

        mapped_values_mask = ['{0:.1E}'.format(p) in sweep_values_str for p in param_values]
        df['plotted'] = mapped_values_mask
        df = df.sort_values(by=[df_cols[0]])
        df_selection = df[df['plotted'] == True]
        # print(df_selection)

        # for sv, tv in zip(sweep_values_str, mapped_values_mask):
        #     print('sv = {0}, tv = {1}'.format(sv,tv))

        # values = df[df.iloc[:, 0].isin(fi['sweep_values'])].sort_values(by=[df_cols[0]])
        # print(values)
        # sweep_values[i] = tuple(values.iloc[:, 1]/3600.)
        df_params = np.array(df_selection.iloc[:, 0])
        df_times = np.array(df_selection.iloc[:, 1]) / 3600
        df_times[np.isinf(df_times)] = 200
        df_max_rsh = np.array(df_selection.iloc[:, 2])
        # simulated_values.sort()
        sweep_values[i] = tuple(df_times)
        max_rsh[i] = tuple(df_max_rsh)
        print(sweep_values[i])

    rects1 = ax1.bar(
        [x - bar_width for x in xticks], sweep_values['low'], color=bar_colors[0], label='Low',
        width=bar_width,
    )
    rects2 = ax1.bar(
        xticks, sweep_values['medium'],
        color=bar_colors[1], label='Medium', width=bar_width,
    )

    rects3 = ax1.bar(
        [x + bar_width for x in xticks], sweep_values['high'],
        color=bar_colors[2], label='High', width=bar_width,
    )

    rects1_r = ax2.bar(
        [x - bar_width for x in xticks], max_rsh['low'], color=bar_colors[0], label='Low',
        width=bar_width,
    )
    rects2_r = ax2.bar(
        xticks, max_rsh['medium'],
        color=bar_colors[1], label='Medium', width=bar_width,
    )

    rects3_r = ax2.bar(
        [x + bar_width for x in xticks], max_rsh['high'],
        color=bar_colors[2], label='High', width=bar_width,
    )

    # ax1.spines['top'].set_visible(False)
    # ax1b.spines['bottom'].set_visible(False)
    #
    # ax1.xaxis.tick_bottom()
    # ax1b.xaxis.tick_top()
    # ax1b.tick_params(labeltop=False)

    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax1b.transAxes, color='k', clip_on=False)
    # ax1b.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    # ax1b.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    #
    # kwargs.update(transform=ax1.transAxes)  # switch to the bottom axes
    # ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xtick_labels)

    ax1.yaxis.set_major_formatter(xfmt)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='upper'))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ### MAX RSH PLOT
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xtick_labels)

    # ax1.yaxis.set_major_formatter(xfmt)
    # ax1.yaxis.set_major_locator(mticker.MaxNLocator(5, prune='upper'))
    # ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # **** BASE CASE *****
    
    ax1.axhline(y=18.35568902, ls='--', lw=1.0, color=(0.8, 0.8, 0.8), zorder=0)
    ax1.annotate(
        "", xy=(-bar_width, 60), xytext=(-bar_width, 40),
        arrowprops=dict(arrowstyle="->", color="tab:green")
    )

    ax2.axhline(y=157.65441577, ls='--', lw=1.0, color=(0.8, 0.8, 0.8), zorder=0)

    # ax1b.yaxis.set_major_formatter(xfmt)
    # ax1b.yaxis.set_major_locator(mticker.MaxNLocator(4, prune='lower'))
    # ax1b.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # fig.text(
    #     0.0, 0.5, '5% Failure Time (h)', va='center', rotation='vertical',
    #     fontsize=12
    # )

    ax1.set_ylabel(r'Time at 1000 $\Omega \cdot \mathregular{cm^2}$ (h)')
    ax1.set_xlabel('Simulation parameter')

    ax2.set_ylabel(r'$R_{\mathrm{sh}}$ at 96 h ($\Omega \cdot $ cm$^{2}$)')
    ax2.set_xlabel('Simulation parameter')

    autolabel(rects1, data_files, 0, "center")
    autolabel(rects2, data_files, 1, "center")
    autolabel(rects3, data_files, 2, "center")

    autolabel_rsh(rects1_r, data_files, 0, "center")
    autolabel_rsh(rects2_r, data_files, 1, "center")
    autolabel_rsh(rects3_r, data_files, 2, "center")
    
    fig.tight_layout()
    fig_mr.tight_layout()
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(results_folder, 'failure_time_85C_summary.svg'), dpi=600)
    fig_mr.savefig(os.path.join(results_folder, '96h_rsh_85C_summary.svg'), dpi=600)
