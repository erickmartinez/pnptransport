# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 16:53:35 2020

@author: Erick
"""

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cmap
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import platform
import os
import matplotlib.gridspec as gridspec
from scipy import interpolate
import itertools
import pidsim.ml_simulator as pmpp_rf

base_folder = r'G:\My Drive\Research\PVRD1\Sentaurus_DDD\pnp_simulations\Cs_1E16\1MVcm'
t_max = 96
transport_simulation_results = [
    r'two_layers_D1=4E-16cm2ps_D2=1E-18cm2ps_Cs1E+16cm3_T85_time96hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp',
    r'two_layers_D1=4E-16cm2ps_D2=1E-17cm2ps_Cs1E+16cm3_T85_time96hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp',
    r'two_layers_D1=4E-16cm2ps_D2=1E-16cm2ps_Cs1E+16cm3_T85_time96hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp',
    r'two_layers_D1=4E-16cm2ps_D2=1E-15cm2ps_Cs1E+16cm3_T85_time96hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp',
    r'two_layers_D1=4E-16cm2ps_D2=1E-14cm2ps_Cs1E+16cm3_T85_time96hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp'
]

pid_experiment_csv = None  # 'G:\My Drive\Research\PVRD1\DATA\PID\MC4_Raw_IV_modified.csv'

sweep_variable = r'$D_{\mathrm{Si}}$ (cm$\mathregular{^2}$/s)'
sweep_variable_units = r'(cm$\mathregular{^{2}}/s$)'
swee_variable_name = r'$D_{\mathrm{Si}}$'
sweep = [1E-18, 1E-17, 1E-16, 1E-15, 1E-14]
sweep_log = True

results_folder = r'G:\My Drive\Research\PVRD1\Manuscripts\thesis\images'

literature_files = [
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Masuda2016_Fig5.csv',
        'time_units': 'min',
        'label': 'Masuda 2016',
        'color': 'tab:red',
        'marker': 'o',
        'type': 'power',
        'normalized': True
    },
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Hacke_ProgressInPhoto2013_Fig1_600V_85C_type2_1.csv',
        'time_units': 'h',
        'label': 'Hacke 2013 type2',
        'color': 'tab:orange',
        'marker': 's',
        'type': 'power',
        'normalized': True
    },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Lausch_IEEEJPV_2014_600V_Rsh.csv',
    #     'time_units': 'h',
    #     'label': 'Lausch 2014',
    #     'color': 'tab:red',
    #     'marker': '^',
    #     'type': 'Rsh',
    #     'normalized': False
    # },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Shutze2011_Fig2_Rsh_high_pid_susceptibility.csv',
    #     'time_units': 'h',
    #     'label': 'Schutze 2011',
    #     'color': 'tab:purple',
    #     'marker': 'o',
    #     'type': 'Rsh',
    #     'normalized': False
    # },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Shutze2011_Fig2_Rsh_medium_pid_susceptibility.csv',
    #     'time_units': 'h',
    #     'label': 'Schutze 2011',
    #     'color': 'tab:purple',
    #     'marker': 'o',
    #     'type': 'Rsh',
    #     'normalized': False
    # },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Bahr_2015_Rsh_normalized_Fig3a_1000_82C.csv',
    #     'time_units': 'h',
    #     'label': 'Bahr 2015',
    #     'color': 'tab:orange',
    #     'marker': 's',
    #     'type': 'Rsh',
    #     'normalized': True
    # },
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Hacke_ProgressInPhoto2013_Fig1_600V_85C_type1.csv',
        'time_units': 'h',
        'label': 'Hacke 2013',
        'color': 'tab:orange',
        'marker': 's',
        'type': 'power',
        'normalized': True
    },
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Oh-MicroelectronicsReliability_2017-Fig1_Pmax_1000V_85C_85RH.csv',
        'time_units': 'h',
        'label': 'Oh 2017',
        'color': 'tab:orange',
        'marker': 's',
        'type': 'power',
        'normalized': True
    },
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Pingel_2010_Normalized_power_1000V_edited.csv',
        'time_units': 'h',
        'label': 'Pingel 2010',
        'color': 'tab:orange',
        'marker': 's',
        'type': 'power',
        'normalized': True
    },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Shutze2011_Fig2_Rsh_low_pid_susceptibility.csv',
    #     'time_units': 'h',
    #     'label': 'Schutze 2011',
    #     'color': 'tab:purple',
    #     'marker': 'o',
    #     'type': 'Rsh',
    #     'normalized': False
    # },
    # {
    #     'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Islam_RenewableEnergy2018_Fig10_Rsh_1000V.csv',
    #     'time_units': 'h',
    #     'label': 'Islam 2018',
    #     'color': 'tab:orange',
    #     'marker': 's',
    #     'type': 'Rsh',
    #     'normalized': False
    # },
    {
        'file': r'G:\My Drive\Research\PVRD1\Literature\PID_degradation_time\Hacke_IEEEJPV_2015_Fig1_Pmax_85C85RH_1.csv',
        'time_units': 'h',
        'label': 'Hacke 2015',
        'color': 'tab:orange',
        'marker': 's',
        'type': 'power',
        'normalized': True
    },
]



defaultPlotStyle = {
    'font.size': 11,
    'font.family': 'Arial',
    'font.weight': 'regular',
    'legend.fontsize': 12,
    'mathtext.fontset': 'stix',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4.5,
    'xtick.major.width': 1.75,
    'ytick.major.size': 4.5,
    'ytick.major.width': 1.75,
    'xtick.minor.size': 2.75,
    'xtick.minor.width': 1.0,
    'ytick.minor.size': 2.75,
    'ytick.minor.width': 1.0,
    'xtick.top': False,
    'ytick.right': False,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'lines.markeredgewidth': 0.85,
    'axes.labelpad': 5.0,
    'axes.labelsize': 12,
    'axes.labelweight': 'regular',
    'legend.handletextpad': 0.2,
    'legend.borderaxespad': 0.2,
    'axes.linewidth': 1.25,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 6,
    'figure.titleweight': 'bold',
    'figure.dpi': 100
}

# From previous run
# failure_times = np.array([ 4.82853982e-01,  1.00895957e+01,  1.36035176e+00,  8.49797337e+01,
#         1.25741181e+00,  9.20011859e+00,  1.06763819e+00,  1.31896482e+00,
#         1.88442211e+00, -1.34000000e-03,  1.20782544e+00,  7.60653266e-01])

if __name__ == '__main__':
    if platform.system() == 'Windows':
        base_folder = r'\\?\\' + base_folder
        if pid_experiment_csv is not None:
            pid_experiment_csv = r'\\?\\' + pid_experiment_csv

    # If an experimental profile is provided load the csv
    if pid_experiment_csv is not None:
        pid_experiment_df = pd.read_csv(pid_experiment_csv)

    mpl.rcParams.update(defaultPlotStyle)
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))
    
    fig_p = plt.figure(1)
    fig_p.set_size_inches(6.5, 3.5, forward=True)
    fig_p.subplots_adjust(hspace=0.1, wspace=0.35)
    gs0_p = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_p, width_ratios=[1])
    gs00_p = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0_p[0])
    ax1_p = fig_p.add_subplot(gs00_p[0, 0])

    # fig_r = plt.figure(2)
    # fig_r.set_size_inches(5.5, 3.5, forward=True)
    # fig_r.subplots_adjust(hspace=0.1, wspace=0.35)
    # gs0_r = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_r, width_ratios=[1])
    # gs00_r = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0_r[0])
    # ax1_r = fig_r.add_subplot(gs00_r[0, 0])

    cm = cmap.get_cmap('rainbow_r')

    
    n_plots = len(literature_files)
    normalize = mpl.colors.Normalize(vmin=0, vmax=n_plots)
    plot_colors = [cm(normalize(i)) for i in range(n_plots)]
    plot_marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p'))
    dashes = [1, 1, 2, 1]# 10 points on, 5 off, 100 on, 5 off

    zorder = n_plots
    t_max = -1
    fail_time_5 = np.empty(n_plots, dtype=np.float)
    for i, lf in enumerate(literature_files):
        k = n_plots - i
        fn = lf['file']
        time_units = lf['time_units']
        label = lf['label']
        if lf['type'] == 'power':
            column_names = ['time', 'power']
        else:
            column_names = ['time', 'Rsh']
        lit_df = pd.read_csv(fn, skiprows=0, header=0, names=column_names, index_col=False)
        
        if time_units == 'min':
            time_lf = lit_df['time'].to_numpy()/60
        elif time_units == 's':
            time_lf = lit_df['time'].to_numpy()/3600
        elif time_units == 'h':
            time_lf = lit_df['time'].to_numpy()
        else:  # Assume hours
            time_lf = lit_df['time'].to_numpy()
            
        t_max = max(time_lf.max(), t_max)
        
        if lf['type'] == 'power':
            normalized_power = lit_df['power']
            ax1_p.plot(
                time_lf, normalized_power, fillstyle='none', 
                color=plot_colors[i], label=label,
                marker=next(plot_marker), zorder=zorder,
                dashes=dashes, lw=1.75
            )
            
            t_interp = np.linspace(np.amin(time_lf), np.amax(time_lf), num=200)
            f_p_interp = interpolate.interp1d(time_lf, normalized_power, kind='linear')
            metric_interp = f_p_interp(t_interp)
            idx_5 = (np.abs(metric_interp - 0.95)).argmin()
            fail_time_5[i] = t_interp[idx_5]
            
        else:
            rsh = lit_df['Rsh']
            if not lf['normalized']:
                rsh = rsh/rsh[0]
            # ax1_r.plot(
            #     time_lf, rsh, fillstyle='none',
            #     color=plot_colors[i], label=label,
            #     marker=next(plot_marker), zorder=k,
            #     ls='--'
            # )

            t_interp = np.linspace(np.amin(time_lf), np.amax(time_lf), num=200)
            f_p_interp = interpolate.interp1d(time_lf, rsh, kind='linear')

            metric_interp = f_p_interp(t_interp)
            idx_5 = (np.abs(metric_interp - 0.95)).argmin()
            fail_time_5[i] = t_interp[idx_5]
            
        zorder += 1

    failure_times = np.empty(
        len(transport_simulation_results), dtype=np.dtype([
            ('D (cm^2/s)', 'd'),
            ('t 5% loss (s)', 'd'),
            # ('t 10% loss (h)', 'd'),
            # ('t 15% loss (h)', 'd'),
            # ('t 20% loss (h)', 'd'),
        ])
    )

    c_map1 = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.LogNorm(vmin=min(sweep), vmax=max(sweep))

    for i, fn, sv in zip(range(len(transport_simulation_results)), transport_simulation_results, sweep):
        # rsh_analysis = prsh.Rsh(h5_transport_file=path_to_h5)
        ml_simulation = pmpp_rf.MLSim(h5_transport_file=os.path.join(base_folder, fn + '.h5'))
        time_s = ml_simulation.time_s
        time_h = time_s / 3600.
        requested_indices = ml_simulation.get_requested_time_indices(time_s)
        pmpp = ml_simulation.pmpp_time_series(requested_indices=requested_indices)
        rsh = ml_simulation.rsh_time_series(requested_indices=requested_indices)

        t_interp = np.linspace(np.amin(time_s), np.amax(time_s), num=200)
        f_p_interp = interpolate.interp1d(time_s, pmpp, kind='linear')
        pmpp_interp = f_p_interp(t_interp)

        idx_5 = (np.abs(pmpp_interp / pmpp_interp[0] - 0.95)).argmin()
        # idx_10 = (np.abs(pmpp_interp / pmpp_interp[0] - 0.9)).argmin()
        # idx_15 = (np.abs(pmpp_interp / pmpp_interp[0] - 0.85)).argmin()
        # idx_20 = (np.abs(pmpp_interp / pmpp_interp[0] - 0.8)).argmin()

        failure_times[i] = (
            sv,
            t_interp[idx_5], #t_interp[idx_10], t_interp[idx_15],
            #t_interp[idx_20],
        )

        sv_str = '{0:.1E}'.format(sv)
        sv_arr = sv_str.split('E')
        sv_arr = np.array(sv_arr, dtype=float)
        sv_txt = r'{0} = $\mathregular{{ 10^{{{1:.0f}}} }}$ {2}'.format(swee_variable_name, sv_arr[1],
                                                                        sweep_variable_units)

        ax1_p.plot(
            time_h, pmpp / pmpp[0], color=c_map1(normalize(sv)), ls='-',
            zorder=zorder, alpha=1.0, label=sv_txt,
            # marker='o', fillstyle='none'
        )
        # ax1_r.plot(
        #     time_h, rsh / rsh[0], color=c_map1(normalize(sv)), ls='-',
        #     zorder=zorder,  # label=sv_txt
        #     # marker='o', fillstyle='none'
        # )

        zorder += 1

    if pid_experiment_csv is not None:
        time_exp = pid_experiment_df['time (s)'] / 3600.
        pmax_exp = pid_experiment_df['Pmax']
        ax1_p.plot(
            time_exp, pmax_exp / pmax_exp[0], ls='None', marker='s', fillstyle='none', label='85°C 1kV ASU',
            zorder=zorder, color='tab:red'
        )
        zorder += 1
        
    ax1_p.set_ylabel('Normalized Power')
    # ax1_r.set_ylabel('Normalized $R_{sh}$')
    
    ax1_p.set_xlim(0, 48)
    # ax1_r.set_xlim(0, 48)
    ax1_p.set_xlabel('Time (hr)')
    # ax1_r.set_xlabel('Time (hr)')
    
    # ax1.tick_params(labelbottom=True, top=False, right=True, which='both', labeltop=False)
    # ax2.tick_params(labelbottom=True, top=False, right=True, which='both')
    
    
    # ax1_r.set_yscale('log')
    # ax1_r.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
    # ax1_r.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1))

    ax1_p.xaxis.set_major_formatter(xfmt)
    ax1_p.xaxis.set_major_locator(mticker.MaxNLocator(7, prune=None))
    ax1_p.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1_p.yaxis.set_major_formatter(xfmt)
    ax1_p.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    ax1_p.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # ax1_r.xaxis.set_major_formatter(xfmt)
    # ax1_r.xaxis.set_major_locator(mticker.MaxNLocator(7, prune=None))
    # ax1_r.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    
    leg1 = ax1_p.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', borderaxespad=0., ncol=1, frameon=False)
    # leg2 = ax1_r.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', borderaxespad=0., ncol=1, frameon=False)
    
    plt.tight_layout()
    fig_p.savefig(os.path.join(results_folder, 'simulated_failure_time_vs_literature_power.svg'), dpi=600)
    fig_p.savefig(os.path.join(results_folder, 'simulated_failure_time_vs_literature_power.png'), dpi=600)
    # fig_r.savefig(os.path.join(results_folder, 'simulated_failure_time_vs_literature_rsh.svg'), dpi=600)
    # fig_r.savefig(os.path.join(results_folder, 'simulated_failure_time_vs_literature_rsh.png'), dpi=600)
    print('Mean failure time: {0:.3f} h'.format(fail_time_5.mean()))

    plt.show()

    df_degradation = pd.DataFrame(failure_times)
    df_degradation.to_csv(
        path_or_buf=os.path.join(
            results_folder,
            'failure_time_0.5MVcm_85C_ML.csv'
        ),
        index=False
    )