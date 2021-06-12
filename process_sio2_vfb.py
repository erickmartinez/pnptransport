import numpy as np
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from scipy import constants
import h5py
import os
import pnptransport.utils as utils
# import shutil
import itertools
import pandas as pd

base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations\SiO2_figure'
output_file = r'sio2_vfb_different_temperatures.h5'
pnp_file = r'G:\My Drive\Research\PVRD1\Manuscripts\PNP_Draft\simulations\SiO2_figure\MU3_SiOx_CV2_VS_1C_D69D70D71D72_4Na_D69D70D71D72_smoothed_cont_minus_clean_5_test1.h5'

er = 3.9
thickness = 100E-7  # cm
experimental_files = [
    {
        'filename': r'50C_MU2_SiOx_CV2_VS_1C_D29D30D31D32_4Na_D29D30D31D32 - Copy_smoothed_cont_minus_clean_5_erf_fit.h5',
        'temperature': 50.,
        'sqrt_t_sat': 2.4
    },
    {
        'filename': r'55C_MU3_SiOx_CV2_VS_1C_D93D94D95D96_4Na_D93D94D95D96 - Copy_smoothed_cont_minus_clean_8_erf_fit.h5',
        'temperature': 55.,
        'sqrt_t_sat': 2.0
    },
    {
        'filename': r'60C_MU3_SiOx_CV2_VS_1C_D69D70D71D72_4Na_D69D70D71D72_smoothed_cont_minus_clean_5_erf_fit.h5',
        'temperature': 60,
        'sqrt_t_sat': 2.84
    },
    {
        'filename': r'70C_fast_MU3_SiOx_CV2_VS_1C_D57D58D59D60_4Na_D57D58D59D60_smoothed_cont_minus_clean_8_erf_fit.h5',
        'temperature': 70,
        'sqrt_t_sat': 1.7
    },
    # {
    #     'filename': r'70C_slow_MU3_SiOx_CV2_VS_1C_D57D58D59D60_4Na_D57D58D59D60_smoothed_cont_minus_clean_7_erf_fit.h5',
    #     'temperature': 70
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

vfb_dtype_2 = np.dtype([
    ('time_s', 'd'),
    ('vsh', 'd'),
    ('vsh_std', 'd'),
    ('Qs', 'd'),
    ('Q0', 'd')
])


def normalize_vfb(v_fb: np.ndarray) -> np.ndarray:
    return -(v_fb - v_fb[0]) / (v_fb.max() - v_fb[0])


def estimate_vfb_norm_error_factor(
        v_fb: np.ndarray, v_fb_std: np.ndarray, v_fb_sat: float, v_fb_sat_std: float
) -> np.ndarray:
    x = v_fb
    x_sat = v_fb_sat
    dx = v_fb_std
    dx_sat = v_fb_sat_std
    factor = np.empty_like(v_fb)
    for i in range(len(x)):
        xx = x[i] if i > 0 else x[1]
        var = np.array([dx[i] / xx, dx_sat / x_sat])
        factor[i] = np.sqrt(var.T.dot(var))
    return factor


marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p', 's', ',', '+', '.', '*'))

if __name__ == '__main__':
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path

    # Load the style
    mpl.rcParams.update(defaultPlotStyle)

    n_files = len(experimental_files)
    cmap = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.Normalize(vmin=0, vmax=n_files)
    plot_colors = [cmap(normalize(i)) for i in range(n_files)]

    fig_s = plt.figure(1)
    fig_s.set_size_inches(7.2, 3.0, forward=True)
    fig_s.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_s_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_s)
    gs_s_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=2, subplot_spec=gs_s_0[0], hspace=0.1, wspace=0.6
    )

    ax_c_0 = fig_s.add_subplot(gs_s_00[0, 0])
    # Set the axis labels
    ax_c_0.set_xlabel(r'Depth ($\mathregular{\mu}$m)')
    # ax_c_0.set_ylabel(r'${C}$ ($\mathregular{cm^{-3}}$)')
    ax_c_0.set_ylabel(r'${C/C_0}$')
    ax_c_0.set_title('(a)')

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
        normalize = mpl.colors.Normalize(vmin=1E-3, vmax=(np.amax(time_s / 3600)))
        # Get a 20 time points geometrically spaced
        requested_time = utils.geometric_series_spaced(max_val=np.amax(time_s), min_delta=240, steps=15)
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
                color_j = cm(normalize(time_j))
                ax_c_0.plot(x_sin, c_sin/c0, color=color_j, zorder=0)
                pbar.set_description('Extracting profile {0} at time {1:.1f} h...'.format(ct_ds, time_j))
                pbar.update()
                pbar.refresh()

    # Set the limits for the x axis of the concentration plot
    ax_c_0.set_xlim(left=np.amin(x_sin), right=np.amax(x_sin))
    # Add the color bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax_c_0)
    cax = divider.append_axes("right", size="7.5%", pad=0.03)
    cbar = fig_s.colorbar(scalar_maps, cax=cax)
    cbar.set_label(r'$t$ (h)', rotation=90, fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.yaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))

    ax_s_0 = fig_s.add_subplot(gs_s_00[0, 1])
    # Set the axis labels
    # ax_s_0.set_xlabel(r'$\sqrt{t}$ (h$^{1/2}$)')
    # ax_s_0.set_ylabel(r"$Q'_s/q$ ($10^{11}/\mathregular{cm}^2$)")
    ax_s_0.set_xlabel(r'$t$ (h)')
    ax_s_0.set_ylabel(r"$Q_s/Q_0$")
    ax_s_0.set_title('(b)')

    # Configure the ticks for the x axis
    ax_s_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_s_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # Create the output storage file
    output_h5 = os.path.join(base_path, output_file)
    if os.path.exists(output_h5):
        os.remove(output_h5)

    # A dataset with all the vfb data for different temperatures (assuming vfb was
    # taken at the same times)
    vfb_df = pd.DataFrame()
    average_vfb_norm_cols = []
    average_vfb_std_norm_cols = []

    for i, f in enumerate(experimental_files):
        fn = f['filename']
        temperature = f['temperature']
        full_path = os.path.join(base_path, fn)
        print("Trying to read file \n'{0}'.".format(full_path))
        with h5py.File(full_path, 'r') as hf_exp:
            exp_ds = np.array(hf_exp['/vfb_data'])
            bias = hf_exp['/vfb_data'].attrs['stress_bias']
            temp_c = hf_exp['/vfb_data'].attrs['temp_c']
            thickness = hf_exp['/vfb_data'].attrs['thickness']
            time_s = exp_ds['time_s']
            if i == 0:
                vfb_df['time (s)'] = time_s
            # print("Length of vfb dataset: {0:d}".format(len(time_s)))
            # print("Time point: ")
            # print(time_s)

            sqrt_t = np.sqrt(time_s / 3600.)
            Qs = exp_ds['vsh'] * constants.epsilon_0 * er / thickness / 100.
            idx_sat = np.abs(sqrt_t - f['sqrt_t_sat']).argmin()
            Q0 = np.mean(Qs[sqrt_t >= f['sqrt_t_sat']])
            vsh = exp_ds['vsh']
            vsh_std = exp_ds['vsh_std']
            vfb_sat = np.mean(vsh[sqrt_t >= f['sqrt_t_sat']])
            vfb_sat_std = np.linalg.norm(vsh_std[sqrt_t >= f['sqrt_t_sat']])
            print('vsh_sat = {0:.4f} ± {1:.4f}'.format(vfb_sat, vfb_sat_std))
            vfb_col = 'vsh @ {0}C'.format(temperature)
            vfb_std_col = 'vsh_std @ {0}C'.format(temperature)
            vfb_norm_col = 'vsh_norm @ {0}C'.format(temperature)
            vfb_std_norm_col = 'vsh_norm_std @ {0}C'.format(temperature)
            average_vfb_norm_cols.append(vfb_norm_col)
            average_vfb_std_norm_cols.append(vfb_std_norm_col)
            vfb_df[vfb_col] = exp_ds['vsh']
            vfb_df[vfb_std_col] = exp_ds['vsh_std']
            vfb_norm_std = np.sign(vfb_sat) * exp_ds['vsh'] / vfb_sat
            vfb_df[vfb_norm_col] = vfb_norm_std
            vfb_df[vfb_std_norm_col] = estimate_vfb_norm_error_factor(
                v_fb=exp_ds['vsh'], v_fb_std=exp_ds['vsh_std'], v_fb_sat=vfb_sat,
                v_fb_sat_std=vfb_sat_std
            ) * vfb_norm_std
            vfb_attrs = {}
            for k in list(hf_exp['vfb_data'].attrs):
                vfb_attrs[k] = hf_exp['vfb_data'].attrs[k]

        if os.path.exists(output_h5):
            h5_mode = 'a'
        else:
            h5_mode = 'w'
        with h5py.File(output_h5, h5_mode) as hf_out:
            output_ds = np.empty(len(time_s), dtype=vfb_dtype_2)
            for j in range(len(time_s)):
                output_ds[j] = (exp_ds['time_s'][j], exp_ds['vsh'][j], exp_ds['vsh_std'][j], Qs[j], Q0)
            qs_ds = hf_out.create_dataset(
                name='{0:.0f}C_{1:d}'.format(temperature, i), data=output_ds, compression='gzip'
            )
            # for a in list(vfb_attrs):
            #     print('Saving attribute \'{0}\' = {1}'.format(a, vfb_attrs[a]))
            #     qs_ds.attrs[a] = vfb_attrs[a]

    vfb_df['vfb_norm_temp_mean'] = vfb_df[average_vfb_norm_cols].mean(axis=1)
    vfb_df['vfb_norm_temp_mean_std'] = vfb_df[average_vfb_norm_cols].std(axis=1)
    print(vfb_df)
    # ax_s_0.errorbar(
    #     time_s / 3600., Qs / abs(Q0), yerr=(2 ** 0.5) * exp_ds['vsh_std'], color=plot_colors[i],
    #     marker=next(marker), fillstyle='none', ls='none', capsize=4, elinewidth=1.5,
    #     label='{0:.0f} °C'.format(temperature), zorder=i
    # )
    ax_s_0.errorbar(
        time_s / 3600., np.abs(vfb_df['vfb_norm_temp_mean']), yerr=vfb_df['vfb_norm_temp_mean_std'],
        color='C0', marker=next(marker), fillstyle='none', ls='none', capsize=4,
        elinewidth=1.5, label='Experiment', zorder=0
    )

    # Plot the simulated vfb
    h5_pnp = os.path.join(base_path, pnp_file)
    with h5py.File(h5_pnp, 'r') as hf:
        # Get the time dataset
        time_sim = np.array(hf['time'])
        vfb = -np.array(hf['vfb'])  # / L * 1E4
        ax_s_0.plot(
            time_sim / 3600., (vfb - vfb[0]) / (vfb.max() - vfb[0]), color='tab:red', label='Model',
            zorder=n_files
        )

    leg = ax_s_0.legend(loc='lower right', frameon=True)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(base_path, 'qs_sio2_50-70C.png'), dpi=600)
    fig_s.savefig(os.path.join(base_path, 'qs_sio2_50-70C.svg'), dpi=600)
    fig_s.savefig(os.path.join(base_path, 'qs_sio2_50-70C.eps'), dpi=600)
    plt.show()
