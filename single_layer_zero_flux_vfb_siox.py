import numpy as np
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import pnptransport.utils as utils
from tqdm import tqdm
import h5py
import os


base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations'
pnp_file = r'MU3_SiOx_CV2_VS_1C_D69D70D71D72_4Na_D69D70D71D72_smoothed_cont_minus_clean_5_test1.h5'
exp_file = r'MU3_SiOx_CV2_VS_1C_D69D70D71D72_4Na_D69D70D71D72_smoothed_cont_minus_clean_5_erf_fit.h5'


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
        base_path = r'\\?\\' + base_path

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
    ax_s_0.set_xlabel(r'$\sqrt{t}$ (h$\mathregular{^{1/2}}$)')
    ax_s_0.set_ylabel(r'$\Delta V_{\mathrm{FB}}$ (V)')

    ax_s_0.set_ylim(-0.75, 0.1)

    # Set the ticks for the x axis
    ax_s_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_s_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    # Configure the ticks for the y axis
    ax_s_0.yaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_s_0.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Plot experimental data
    h5_exp = os.path.join(base_path, exp_file)
    with h5py.File(h5_exp, 'r') as hf:
        exp_ds = np.array(hf['vfb_data'])
        bias = hf['vfb_data'].attrs['stress_bias']
        temp_c = hf['vfb_data'].attrs['temp_c']
        thickness = hf['vfb_data'].attrs['thickness']
        print(thickness)
        ax_s_0.errorbar(
            x=np.sqrt(exp_ds['time_s'] / 3600), y=exp_ds['vsh'], yerr=exp_ds['vsh_std'], color='tab:blue', marker='o',
            fillstyle='none', capsize=4, ls='none', label='60 °C, 4 V' #ecolor='k', markeredgecolor='k',
        )

    # Plot the simulated vfb
    h5_pnp = os.path.join(base_path, pnp_file)
    with h5py.File(h5_pnp, 'r') as hf:
        # Get the time dataset
        time_s = np.array(hf['time'])
        vfb = - np.array(hf['vfb'])  # / L * 1E4
        ax_s_0.plot(
            np.sqrt(time_s / 3600.), -(vfb-vfb[0]), color='tab:red', label='Model'
        )
    leg = ax_s_0.legend(loc='upper right', frameon=True)
    ax_s_0.set_xlim(0, np.sqrt(np.amax(time_s/3600.)))
    fig_s.tight_layout()

    filetag = os.path.basename(pnp_file)
    fig_s.savefig(os.path.join(base_path, filetag + '_shift_s.svg'), dpi=600)
    fig_s.savefig(os.path.join(base_path, filetag + '_shift_s.eps'), dpi=600)

    plt.show()