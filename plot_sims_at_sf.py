import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import platform
import h5py

base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\combinatorial'
exp_pid_csv = r'G:\My Drive\Research\PVRD1\DATA\PID\PID_mc_BSF_4_ready.csv'
csv_summary = 'PID_mc_BSF_4_ready_20201231-010103.h5_grid.csv'
csv_sims = r'G:\My Drive\Research\PVRD1\DATA\PID\201217Data_SIMS_at_SF_1_and_2.csv'
FWHM_nm = 15
CONST = 2.0 * np.sqrt(2.0 * np.log(2.0))
sims_center =  50.0


def gaussian(fwhm, x0, x_, a=1.0):
    s = fwhm / CONST
    s2 = s ** 2.0
    return a*np.exp(-((x_ - x0) ** 2.0) / (2.0 * s2)) #/ (np.sqrt( 2. * np.pi) * s)


def cost_function(y, rsh) -> float:
    y = np.array(y)
    m = len(y)
    diff = y / y[0] - rsh / rsh[0]
    r = 0.5 * np.dot(diff.T, diff) / m
    return r


if __name__ == '__main__':
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path
        exp_pid_csv = r'\\?\\' + exp_pid_csv
        csv_sims = r'\\?\\' + csv_sims
    # Load my style
    with open('plotstyle.json', 'r') as style_file:
        mpl.rcParams.update(json.load(style_file)['defaultPlotStyle'])
    # load the experimental pid csv
    exp_df = pd.read_csv(exp_pid_csv)
    exp_df = exp_df[exp_df['time (s)'] <= 345600]
    time = np.array(exp_df['time (s)'].values)
    rsh = np.array(exp_df['Rsh (ohm cm^2)'].values)
    rsh_norm = rsh / rsh[0]
    # load the summary csv
    grid_df = pd.read_csv(os.path.join(base_path, csv_summary))
    n_simulations = len(grid_df)
    costs = np.empty(n_simulations, dtype=np.float)
    file_tag = os.path.splitext(csv_summary)[0]

    for i, r in grid_df.iterrows():
        csv_file = os.path.splitext(os.path.basename(r['h5 file']))[0] + '.csv'
        csv_dir = os.path.basename(os.path.dirname(r['h5 file']))
        csv_file = os.path.join(base_path, 'combinations', csv_dir, csv_file)
        print(csv_file)
        df = pd.read_csv(csv_file)
        y = df['Rsh norm'].values
        costs[i] = cost_function(y=y, rsh=rsh)
    grid_df['cost'] = costs
    grid_df.to_csv(path_or_buf=csv_summary)

    idx_min = np.argmin(costs)
    opt_h5 = grid_df.iloc[idx_min]['h5 file']
    rpath = os.path.join(base_path, 'combinations', os.path.basename(os.path.dirname(opt_h5)))
    opt_basename = os.path.splitext(os.path.basename(opt_h5))[0]
    opt_csv = os.path.join(rpath, opt_basename + '.csv')
    opt_df = pd.read_csv(opt_csv)

    print('Optimum csv:')
    print(opt_csv)
    print('Cost: {0:.4g}'.format(grid_df.iloc[idx_min]['cost']))
    print('S0: {0:.3E} (1/cm^2), h: {1:.3E} (cm/s), DSF: {2:.3E} (cm^2/s)'.format(
        grid_df.iloc[idx_min]['S0 (1/cm^2)'],
        grid_df.iloc[idx_min]['h (cm/s)'],
        grid_df.iloc[idx_min]['D_SF (cm^2/s)']
    ))

    # Plot PID data
    fig_pid = plt.figure(1)
    fig_pid.set_size_inches(4.5, 3.0, forward=True)
    fig_pid.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_pid)
    gs_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs_0[0], hspace=0.1,
    )
    ax_pid = fig_pid.add_subplot(gs_00[0, 0])
    ax_pid.set_xlabel('Time (hr)')
    ax_pid.set_ylabel('Normalized $R_{\mathrm{sh}}$')
    ax_pid.set_xlim(0, 96.)

    idx = time < 96. * 3600.

    ax_pid.plot(
        time / 3600, rsh_norm, marker='o', ls='none', fillstyle='none', label='Experiment',
        color='k'
    )
    ax_pid.plot(
        time / 3600, opt_df['Rsh norm'].values, ls='-', color='tab:red',
        label='Best fit'
    )

    ax_pid.set_ylim(1E-2, 2.0)

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    ax_pid.xaxis.set_major_formatter(xfmt)
    ax_pid.xaxis.set_major_locator(mticker.MaxNLocator(12, prune=None))
    ax_pid.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax_pid.set_yscale('log')

    leg = ax_pid.legend(loc='lower right', frameon=True)
    opt_fig_tag = os.path.join(base_path, file_tag + '_optimized')

    fig_pid.tight_layout()

    fig_pid.savefig(opt_fig_tag + '.png', dpi=300)
    fig_pid.savefig(opt_fig_tag + '.svg', dpi=600)
    fig_pid.savefig(opt_fig_tag + '.eps', dpi=600)

    # Plot SIMS data
    opt_h5 = grid_df.iloc[idx_min]['h5 file']
    rpath = os.path.join(base_path, 'combinations', os.path.basename(os.path.dirname(opt_h5)))
    opt_basename = os.path.basename(opt_h5)
    opt_h5 = os.path.join(rpath, opt_basename)

    from scipy import interpolate

    with h5py.File(opt_h5, 'r') as hf:
        time_sims = np.array(hf['/time'])
        n_profiles = len(time_sims)
        # Get the sinx group
        grp_sinx = hf['L1']
        # get the si group
        grp_si = hf['L2']
        # Get the position vector in SiNx in nm
        x_sin = np.array(grp_sinx['x']) * 1000.
        x_si = np.array(grp_si['x']) * 1000.
        # Get the specific profile
        idx_max = n_profiles - 1
        ct_ds = 'ct_{0:d}'.format(idx_max)
        c_sin = np.array(grp_sinx['concentration'][ct_ds])
        c_si = np.array(grp_si['concentration'][ct_ds])
        x_model = np.hstack((x_sin, x_si))
        c_model = np.hstack((c_sin, c_si))

        f = interpolate.interp1d(x_model, c_model)
        x = np.linspace(x_model.min(), x_model.max(), 2000)
        c = f(x)

    sigma = FWHM_nm / CONST
    n_points = len(x)
    # print('LENGTH OF MODEL VECTOR: {0}'.format(n_points))
    print('sigma = {0:.3f} nm'.format(sigma))
    idx_6_sigma = np.abs(x - 75.0) <= 3.0 * sigma
    x_6_sigma = x[idx_6_sigma]
    print('x between 6 sigma:')
    print(x_6_sigma)
    print('Number of points in 6 sigma: {0}'.format(len(x_6_sigma)))
    print('DELTA 6 SIGMA: {0:.3f} nm'.format(6*sigma))
    # x = x - 25.0
    # idx = x > 0
    # x = x[idx]
    # c = c[idx]
    # c = c / c.max()
    center = x_sin.max()

    truncate_at = 3.0 * sigma
    dx = (x.max() - x.min()) / n_points
    print('dx = {0:.3g}'.format(dx))

    # x_g = np.linspace(center - 6.0 * sigma, center + 6.0 * sigma, 100)
    x_g = np.arange(center - 6.0 * sigma, center + 6.0 * sigma + dx, dx)
    # print('x_g:')
    # print(x_g)
    x_p = np.linspace(sims_center - 6.0 * sigma, sims_center + 6.0 * sigma, 100)
    g_p = gaussian(fwhm=FWHM_nm, x0=sims_center, x_=x_p, a=0.5)
    y_g = gaussian(fwhm=FWHM_nm, x0=center, x_=x_g, a=0.5)
    y_g[x_g < (center - truncate_at)] = 0.0
    y_g[x_g > (center + truncate_at)] = 0.0
    g_p[x_p < (sims_center - truncate_at)] = 0.0
    g_p[x_p > (sims_center + truncate_at)] = 0.0
    y_c = np.convolve(c, y_g, mode="valid")
    x_c = np.linspace(x.min(), x.max(), len(y_c))

    # Load SIMS experimental data
    exp_sims_df = pd.read_csv(filepath_or_buffer=csv_sims).dropna()
    x1 = exp_sims_df['S1 Depth (nm)'].values
    x2 = exp_sims_df['S2 Depth (nm)'].values
    c1 = exp_sims_df['S1 Na (atoms/cm3)'].values
    c2 = exp_sims_df['S2 Na (atoms/cm3)'].values

    # print(exp_sims_df)

    fig_sims = plt.figure(2)
    fig_sims.set_size_inches(4.5, 3.0, forward=True)
    fig_sims.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_sims_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_sims)
    gs_sims_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs_sims_0[0], hspace=0.1,
    )

    ax_sims = fig_sims.add_subplot(gs_sims_00[0, 0])
    ax_sims.set_xlabel('Depth (nm)')
    ax_sims.set_ylabel(r'$C_{\mathrm{Na}}/C_{\mathrm{max}}$')
    ax_sims.set_xlim(0, 150)
    ax_sims.set_yscale('log')
    ax_sims.set_ylim(1E-3, 10.0)
    ax_sims.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
    ax_sims.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=60, subs=np.arange(2, 10) * .1))

    ax_sims.plot(
        x1, c1/c1.max(), ls='None', color='tab:grey', label='SIMS 1', marker='o', fillstyle='none'
    )

    ax_sims.plot(
        x2, c2/c2.max(), ls='None', color='k', label='SIMS 2', marker='s', fillstyle='none'
    )

    ax_sims_twin = ax_sims.twinx()
    color = 'tab:red'
    ax_sims_twin.set_ylabel(r'Gaussian thickness distribution', color='b')

    # ax_sims.plot(
    #     x, c / c.max(), ls='-', color=color, label='Model'
    # )

    ax_sims.plot(
        x_c, y_c / y_c.max(), ls='--', color=color, label='Model+Convolution'
    )

    ax_sims_twin.plot(
        x_p, g_p, ls=':', lw=1.0, color='b'
    )
    ax_sims_twin.tick_params(axis='y', labelcolor='b')
    ax_sims_twin.set_ylim(0, 1.0)
    # ax_sims_twin.set_yscale('log')
    # ax_sims_twin.set_ylim(1E10, 1E18)
    # ax_sims_twin.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
    # ax_sims_twin.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1))
    ax_sims_twin.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_sims_twin.axvline(x=sims_center, ls='--', color='k', lw=0.75)

    leg = ax_sims.legend(loc='best', frameon=True, fontsize=10)
    sims_fig_tag = os.path.join(base_path, file_tag + '_sims')

    # Labels to the layers
    ax_sims.text(
        0.025, 0.9,
        r'$\mathregular{SiN_x}$',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax_sims.transAxes,
        fontsize=11,
        fontweight='regular',
        color='tab:green'
    )

    ax_sims.text(
        0.35, 0.92,
        r'Si',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax_sims.transAxes,
        fontsize=11,
        fontweight='regular',
        color='tab:green'
    )

    fig_sims.tight_layout()

    fig_sims.savefig(sims_fig_tag + '.png', dpi=300)
    fig_sims.savefig(sims_fig_tag + '.svg', dpi=600)
    fig_sims.savefig(sims_fig_tag + '.eps', dpi=600)

    plt.show()
    # print(grid_df)
