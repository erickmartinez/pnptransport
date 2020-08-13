import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import platform
import os
import pidsim.rsh as prsh
import re
import h5py

base_folder = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\results_two_layers\pnp\source_limited\source_limited_4um_Cs1E16_10kVcm\trapping'
transport_file = r'1um_traps_48h_recovery_1E-18M_3.750V_h01E-08_SL_D1=4E-16cm2ps_D2=1E-14cm2ps_Cs1E+16cm3_T85_time12hr_h1.0e-12_m1.0e-02_v3.750_pnp.h5'
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
    mpl.rcParams.update(defaultPlotStyle)
    if platform.system() == 'Windows':
        base_folder = r'\\?\\' + base_folder

    results_folder = os.path.join(base_folder, output_folder)

    # Extract the recovery voltage from the file name
    pattern_recovery_voltage = re.compile('M\_(\d+\.\d{1,3})V')
    match_recovery_voltage = pattern_recovery_voltage.search(transport_file)
    try:
        recovery_voltage = float(match_recovery_voltage.group(1))
    except Exception as e:
        print('Could not extract the recovery voltage from filename.')
        recovery_voltage = None

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    path_to_file = os.path.join(base_folder, transport_file)

    # If source limited try to find h0. if not existent assume 1E16 from simulations prior to incorporation of that
    # parameter to the list of stored values in the h5 file
    if not os.path.exists(path_to_file):
        raise FileNotFoundError('Could not find \'{}\''.format(path_to_file))
    with h5py.File(path_to_file, 'a') as hf:
        if 'cs_0' not in hf['time'].attrs:
            hf['time'].attrs['cs_0'] = 1E16
        if 'c_surface' not in hf['time'].attrs:
            hf['time'].attrs['c_surface'] = 1E11

    rsh_analysis = prsh.Rsh(h5_transport_file=path_to_file)
    time_s = rsh_analysis.time_s
    time_points = len(time_s)
    time_h = time_s / 3600
    t_max = np.amax(time_s)

    requested_indices = rsh_analysis.get_requested_time_indices(time_s)
    rsh = rsh_analysis.resistance_time_series(requested_indices=requested_indices)
    c_data = rsh_analysis.interface_concentrations_time_series(requested_indices=requested_indices)
    j_data = rsh_analysis.dielectric_flux_time_series(requested_indices=requested_indices)

    data_storage = rsh_analysis.h5_storage
    time_metadata = data_storage.get_metadata(group='/time')
    layer_1_metadata = data_storage.get_metadata(group='/L1')
    layer_2_metadata = data_storage.get_metadata(group='/L2')

    source_limited = True

    try:
        h0_str = '{0:.1E}'.format(time_metadata['h0'])
        h0_str_exp = '10^{{{{{0}}}}}'.format(h0_str.split('E')[1])
    except Exception as e:
        print('Could not find h0, assuming constant source simulation.')
        source_limited = False


    h_str = '{0:.1E}'.format(time_metadata['h'])
    h_str_exp = '10^{{{{{0}}}}}'.format(h_str.split('E')[1])

    cs_str = '10^{{16}}'

    if recovery_voltage is not None:
        if source_limited:
            title_str = r'$\mathregular{{V_{{r}} = {0:.3f}\;V,\; h_{{s}} = {1}\;cm/s,\; C_{{s}} = {2}\;cm^{{-3}}}}$'.format(
                recovery_voltage, h0_str_exp, cs_str
            )
        else:
            title_str = r'$\mathregular{{V_{{r}} = {0:.3f}\;V,\; C_{{s}} = {1}\;cm^{{-3}}}}$'.format(
                recovery_voltage, cs_str
            )

    else:
        if source_limited:
            title_str = r'$\mathregular{{h_{{s}} = {0}\;cm/s,\; C_{{s}} = {1}\;cm^{{-3}}}}$'.format(
                h0_str_exp, cs_str
            )
        else:
            title_str = r'$\mathregular{{C_{{s}} = {0}\;cm^{{-3}}}}$'.format(
                cs_str
            )

    fig = plt.figure()
    fig.set_size_inches(4.5, 3.0, forward=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs00[0, 0])

    # ax1.plot(time_h, rsh / rsh[0])
    ax1.plot(time_h, rsh)
    ax1.set_xlim(0, np.amax(time_h))
    ax1.set_yscale('log')
    ax1.set_xlabel('time (h)')
    # ax1.set_ylabel(r'$R_{\mathrm{sh}}/R_{\mathrm{sh}}(t=0)$')
    ax1.set_ylabel('$R_{\mathrm{sh}}\;(\Omega \cdot \mathregular{cm^2})$')

    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    if source_limited:
        n_na = time_metadata['na_0'] / 6.022E23
        plot_txt = '$E_{{\\mathrm{{stress}}}} = {0:.3f}$ MV/cm\n$h={1}$ cm/s\n$N = {2:.3g}$ moles'.format(
            layer_1_metadata['electric_field_app'], h_str_exp, n_na
        )
    else:
        plot_txt = '$E_{{\\mathrm{{stress}}}} = {0:.3f}$ MV/cm\n$h={1}$ cm/s'.format(
            layer_1_metadata['electric_field_app'], h_str_exp
        )

    ax1.set_title(title_str)
    ax1.text(
        0.55, 0.95,
        plot_txt,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax1.transAxes,
        fontsize=11,
        color='b'
    )

    plt.tight_layout()
    file_tag = os.path.splitext(transport_file)[0]

    fig_fn = os.path.join(results_folder, file_tag + '_rsh.png')
    fig.savefig(fig_fn, dpi=600)

    # Save a csv with Rsh time series
    output_data = np.empty(len(rsh), dtype=np.dtype([('time (h)', 'd'), ('Rsh (Ohm cm^2)', 'd')]))
    for i, t, r in zip(range(len(rsh)), time_h, rsh):
        output_data[i] = (t, r)
    df = pd.DataFrame(data=output_data)
    df.to_csv(path_or_buf=os.path.join(results_folder, file_tag + '.csv'), index=False)
    print('Shunt Area: {0:.3E} cm^2'.format(rsh_analysis.shunt_area))
    # plt.show()

    # ******************** Interface concentration plot ***********************
    fig = plt.figure()
    fig.set_size_inches(4.5, 3.0, forward=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs00[0, 0])

    # ax1.plot(time_h, rsh / rsh[0])
    ax1.plot(time_h, c_data['C_SiNx (cm^-3)'], label=r'$C_{\mathrm{SiN_x}}$')
    ax1.plot(time_h, c_data['C_Si (cm^-3)'], label=r'$C_{\mathrm{Si}}$')
    ax1.set_xlim(0, np.amax(time_h))

    leg = ax1.legend(loc='lower right', frameon=False)

    ax1.set_yscale('log')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel(r'$C$ at interface ($\mathregular{cm^{-3}}$)')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1.set_title(title_str)
    ax1.text(
        0.05, 0.05,
        plot_txt,
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax1.transAxes,
        fontsize=11,
        color='b'
    )

    plt.tight_layout()

    fig_fn = os.path.join(results_folder, file_tag + '_cint.png')
    fig.savefig(fig_fn, dpi=600)
    # plt.show()

    # ******************** Dielectric flux plot ***********************
    fig = plt.figure()
    fig.set_size_inches(4.5, 3.5, forward=True)
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[0])
    ax1 = fig.add_subplot(gs00[0, 0])

    # ax1.plot(time_h, rsh / rsh[0])
    ax1.plot(time_h, j_data['J_1 (cm/s)'], label=r'$J(x=0)$')
    ax1.plot(time_h, j_data['J_2 (cm/s)'], label=r'$J(x=L/2)$')
    ax1.set_xlim(0, np.amax(time_h))

    leg = ax1.legend(loc='center right', frameon=False)

    ax1.set_yscale('symlog')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel(r'$J(x=L/2), J(x=0)$')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    s = ax1.yaxis._scale
    locmaj = mpl.ticker.SymmetricalLogLocator(transform=s._transform, subs=s.subs)
    locmaj.set_params(numticks=7)
    locmin = mpl.ticker.SymmetricalLogLocator(transform=s._transform, subs=s.subs)
    locmin.set_params(numticks=14)
    ax1.yaxis.set_major_locator(locmaj)
    ax1.yaxis.set_minor_locator(locmin)

    ax1.set_title(title_str)
    ax1.text(
        0.6, 0.95,
        plot_txt,
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax1.transAxes,
        fontsize=11,
        color='b'
    )

    plt.tight_layout()

    fig_fn = os.path.join(results_folder, file_tag + '_flux.png')
    fig.savefig(fig_fn, dpi=600)
    plt.show()
