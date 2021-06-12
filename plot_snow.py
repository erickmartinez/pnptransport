import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pidsim.ml_simulator as pmpp_rf
import h5py
import os
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import pnptransport.utils as utils
from scipy import constants
from tqdm import tqdm

# base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\thesis\images'
base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations\results_20201031'
# pnp_file = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations\single_layer_zero_flux_10_140C_29E11pcm2_D5.1E-16_L1=0.2um_10V_no_screening_x1=0.015.h5'
pnp_file = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations\results_20201031\single_layer_zero_flux_12h_140C_2.9E10pcm2_D1.11E-15_L1=0.028um_10V.h5'

snow_data_files = [
    {
        'file': 'snow_data_experiment_80C.csv',
        'type': 'experimental',
        'label': '80 °C',
        'marker': '^',
        'fs': 'full',
        'ls': 'None',
        'color': 'C2'
    },
    {
        'file': 'snow_data_experiment_100C.csv',
        'type': 'experimental',
        'label': '100 °C',
        'marker': 'o',
        'fs': 'full',
        'ls': 'None',
        'color': 'C2'
    },
    {
        'file': 'snow_data_experiment_120C.csv',
        'type': 'experimental',
        'label': '120 °C',
        'marker': 's',
        'fs': 'none',
        'ls': 'None',
        'color': 'C2'
    },
    {
        'file': 'snow_data_experiment_140C.csv',
        'type': 'experimental',
        'label': '140 °C',
        'marker': '^',
        'fs': 'none',
        'ls': 'None',
        'color': 'C2'
    },
    {
        'file': 'snow_data_experiment_160C.csv',
        'type': 'experimental',
        'label': '160 °C',
        'marker': 'o',
        'fs': 'none',
        'ls': 'None',
        'color': 'C2'
    },
    # {
    #     'file': 'snow_data_theory_linear.csv',
    #     'type': 'theory',
    #     'label': 'Linear',
    #     'marker': 'None',
    #     'fs': 'none',
    #     'ls': '--',
    #     'color': 'C0'
    # },
    # {
    #     'file': 'snow_data_theory_exp.csv',
    #     'type': 'theory',
    #     'label': 'Exponential',
    #     'marker': 'None',
    #     'fs': 'none',
    #     'ls': '-',
    #     'color': 'C1'
    # }

]

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
        # prepend the paths
        base_path = r'\\?\\' + base_path
        pnp_file = r'\\?\\' + pnp_file

    # Load the style
    mpl.rcParams.update(defaultPlotStyle)

    # Plot Snow data
    fig_s = plt.figure(1)
    fig_s.set_size_inches(4.0, 3.0, forward=True)
    fig_s.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_s_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_s)
    gs_s_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs_s_0[0], hspace=0.1,
    )
    ax_s_0 = fig_s.add_subplot(gs_s_00[0, 0])
    # Set the axis labels
    # ax_s_0.set_xlabel(r'$\sqrt{t}$ (h$\mathregular{^{1/2}}$)')
    ax_s_0.set_xlabel(r'$\sqrt{t/\tau}$')
    ax_s_0.set_ylabel(r'$Q_{\mathrm{S}}/ Q_{0}$')

    ax_s_0.set_ylim(0, 1.1)

    # Set the ticks for the x axis
    ax_s_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_s_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    # Configure the ticks for the y axis
    ax_s_0.yaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_s_0.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    time_max = -1

    for d in snow_data_files:
        fn = d['file']
        # read the file
        qs_df = pd.read_csv(os.path.join(base_path, fn), header=0, skiprows=0, names=['sqrt(t)', 'Qs/Q0'])
        sqrt_time = qs_df['sqrt(t)'].to_numpy()
        qs_q0 = qs_df['Qs/Q0'].to_numpy()
        time_max = max(time_max, np.amax(sqrt_time))
        if d['type'] == 'experimental':
            ax_s_0.plot(
                sqrt_time, qs_q0, color=d['color'], marker=d['marker'], ls=d['ls'], fillstyle=d['fs'],
                label=d['label']
            )
        else:
            ax_s_0.plot(
                sqrt_time, qs_q0, color=d['color'], marker=d['marker'], ls=d['ls'], fillstyle=d['fs'],
            )
    x1 = 0.014
    # Read the h5 file
    with h5py.File(pnp_file, 'r') as hf:
        time_h = np.array(hf['/time']) / 3600.
        try:
            x1 = hf['/time'].attrs['x1']
        except Exception as e:
            x1 = x1
        qs0 = hf['/time'].attrs['surface_concentration']
        D = hf['/L1'].attrs['D']
        L = np.amax(np.array(hf['L1/x']))
        qs = -np.array(hf['QS'])  # / L * 1E4
        vfb = np.array(hf['vfb'])
        tau = 4. * ((x1 * 1E-4 / np.pi) ** 2) / D
        # tau = 1. * ((x1 * 1E-4) ** 2) / D


    # Old data sets
    # ax_s_0.plot(
    #     np.sqrt(time_h * 3600 / tau), np.abs(qs / qs0), color='tab:red', label='This work'
    # )
    # 2020/10 datasets
    ax_s_0.plot(
        np.sqrt(time_h * 3600 / tau), np.abs((qs-qs[0])/(qs[-1]-qs[0])), color='tab:red', label='This work'
    )
    # Set the limits for the x axis
    ax_s_0.set_xlim(left=0, right=time_max)
    leg = ax_s_0.legend(loc='lower right', frameon=True)
    fig_s.tight_layout()
    # plt.show()

    # Concentration profiles
    fig_c = plt.figure(2)
    fig_c.set_size_inches(4.25, 3.0, forward=True)
    fig_c.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_c_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_c)
    gs_c_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs_c_0[0], hspace=0.1,
    )
    ax_c_0 = fig_c.add_subplot(gs_c_00[0, 0])
    # Set the axis labels
    ax_c_0.set_xlabel(r'Depth ($\mathregular{\mu}$m)')
    # ax_c_0.set_ylabel(r'${C}$ ($\mathregular{cm^{-3}}$)')
    ax_c_0.set_ylabel(r'${C/C_0}$')

    # Make the y axis log
    ax_c_0.set_yscale('log')
    # Set the ticks for the y axis
    ax_c_0.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
    ax_c_0.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=60, subs=np.arange(2, 10) * .1))
    # ax_c_0.set_ylim(1E14, 1E20)
    ax_c_0.set_ylim(1E-4, 50.0)
    # Set the ticks for the x axis
    # Configure the ticks for the x axis
    ax_c_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_c_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    color_map = 'viridis_r'
    # Get the color map
    cm = mpl.cm.get_cmap(color_map)

    with h5py.File(pnp_file, 'r') as hf:
        # Get the time dataset
        time_s = np.array(hf['time'])
        # Get the sinx group
        grp_sinx = hf['L1']
        # Get the position vector in SiNx in nm
        x_sin = np.array(grp_sinx['x'])
        thickness_sin = np.max(x_sin)
        n_profiles = len(time_s)
        normalize = mpl.colors.Normalize(vmin=1E-3, vmax=(np.sqrt(np.amax(time_s/tau) )))
        # Get a 20 time points geometrically spaced
        requested_time = utils.geometric_series_spaced(max_val=np.amax(time_s), min_delta=240, steps=20)
        # requested_time = np.linspace(0.0, np.amax(time_s), 50)
        requested_indices = utils.get_indices_at_values(x=time_s, requested_values=requested_time)
        time_profile = np.empty(len(requested_indices))
        C0 = np.array(grp_sinx['concentration']['ct_0'])
        c0 = C0[0]

        print(requested_indices)

        model_colors = [cm(normalize(t)) for t in time_s / 3600.]
        scalar_maps = mpl.cm.ScalarMappable(cmap=cm, norm=normalize)
        with tqdm(requested_indices, leave=True, position=0) as pbar:
            for j, idx in enumerate(requested_indices):
                time_j = time_s[idx] / 3600.
                time_profile[j] = time_j
                # Get the specific profile
                ct_ds = 'ct_{0:d}'.format(idx)
                c_sin = np.array(grp_sinx['concentration'][ct_ds])
                color_j = cm(normalize(np.sqrt(time_j*3600./tau)))
                ax_c_0.plot(x_sin, c_sin/c0, color=color_j, zorder=0)
                pbar.set_description('Extracting profile {0} at time {1:.1f} h...'.format(ct_ds, time_j))
                pbar.update()
                pbar.refresh()

    # Set the limits for the x axis of the concentration plot
    ax_c_0.set_xlim(left=np.amin(x_sin), right=np.amax(x_sin))
    # Add the color bar
    divider = make_axes_locatable(ax_c_0)
    cax = divider.append_axes("right", size="7.5%", pad=0.03)
    cbar = fig_c.colorbar(scalar_maps, cax=cax)
    cbar.set_label(r'$\sqrt{t/\tau}$', rotation=90, fontsize=14)
    cbar.ax.tick_params(labelsize=11)

    # plot_c_sin_txt = source_str1 + '\n' + source_str2
    # ax_c_0.text(
    #     0.95, 0.95,
    #     plot_c_sin_txt,
    #     horizontalalignment='right',
    #     verticalalignment='top',
    #     transform=ax_c_0.transAxes,
    #     fontsize=11,
    #     color='k'
    # )

    fig_c.tight_layout()

    filetag = os.path.basename(pnp_file)
    fig_s.savefig(os.path.join(base_path, filetag + '_s.svg'), dpi=600)
    fig_s.savefig(os.path.join(base_path, filetag + '_s.eps'), dpi=600)
    fig_s.savefig(os.path.join(base_path, filetag + '_s.png'), dpi=600)
    fig_c.savefig(os.path.join(base_path, filetag + '_c.svg'), dpi=600)
    fig_c.savefig(os.path.join(base_path, filetag + '_c.eps'), dpi=600)

    plt.show()
