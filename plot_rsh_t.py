import pidsim.rsh
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
import platform
import os
import matplotlib.gridspec as gridspec
from tqdm import tqdm


base_folder = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\results_two_layers\pnp\source_limited\source_limited_4um_Cs1E16_0.5MVcm\recovery'
t_max = 96
transport_file = r'48h_recovery_two_layers_SL_D1=4E-16cm2ps_D2=1E-14cm2ps_Cs1E+16cm3_T85_time12hr_h1.0e-12_m1.0e+00_v3.750e+00_pnp.h5'
output_folder = r'rsh_analysis'

defaultPlotStyle = {
    'font.size': 11,
    'font.family': 'Arial',
    'font.weight': 'regular',
    'legend.fontsize': 11,
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

if __name__ == '__main__':

    if platform.system() == 'Windows':
        base_folder = r'\\?\\' + base_folder

    results_folder = os.path.join(base_folder, output_folder)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    rsh_analysis = pidsim.rsh.Rsh(h5_transport_file=os.path.join(base_folder, transport_file))
    time_s = rsh_analysis.time_s
    time_points = len(time_s)
    time_h = time_s / 3600
    rsh = np.empty(time_points)
    with tqdm(time_points) as pbar:
        for i, t in enumerate(time_s):
            rsh[i] = rsh_analysis.resistance_at_time_t(time_s=t)
            progress_str = 'Processing Time: {0:.1f}/{1:.1f}, Rsh: {2:.3g} Ohm'.format(t/3600, time_s.max()/3600, rsh[i])
            pbar.set_description(progress_str)

    mpl.rcParams.update(defaultPlotStyle)

    fig = plt.figure()
    fig.set_size_inches(5.5, 4.5, forward=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.35)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs0[0, 0])
    # ax1 = fig.add_subplot(111)
    # ax1.plot(time_h, rsh)
    #
    # plt.tight_layout()
    # plt.show()
