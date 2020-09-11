import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from tqdm import trange
from matplotlib.ticker import ScalarFormatter
import pidsim.parameter_span as pspan
from scipy import interpolate
import pnptransport.utils as utils
import re

base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200831'
span_database = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200831\one_factor_at_a_time_lower_20200828_h=1E-12.csv'
parameter = 'h'
output_folder = 'ofat_comparison'
t_max_h = 48.

parameter_units = {
    'sigma_s': 'cm^{-2}',
    'zeta': 's^{-1}',
    'DSF': 'cm^2/s',
    'E': 'V/cm',
    'm': '',
    'h': 'cm/s',
    'recovery time': 's',
    'recovery electric field': 'V/cm'
}

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


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    Parameters
    ----------
    value: str
        The string

    Returns
    -------
    str
        Normalized string
    """
    value = re.sub('[^\w\s\-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


if __name__ == '__main__':
    output_path = os.path.join(base_path, output_folder)
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path

    output_path = os.path.join(base_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    span_df = pd.read_csv(span_database, index_col=0)
    ofat_df = pd.read_csv(os.path.join(base_path, 'ofat_db.csv'), index_col=None)
    # Get the row corresponding to the values to compare
    parameter_info = span_df.loc[parameter]
    parameter_span = pspan.string_list_to_float(parameter_info['span'])
    units = parameter_units[parameter]
    # Get the values of every other parameter from the ofat_db file
    ofat_constant_parameters = ofat_df.iloc[0]
    time_s = float(ofat_constant_parameters['time (s)'])
    temp_c = float(ofat_constant_parameters['temp (C)'])
    bias = float(ofat_constant_parameters['bias (V)'])
    thickness_sin = float(ofat_constant_parameters['thickness sin (um)'])
    thickness_si = float(ofat_constant_parameters['thickness si (um)'])
    er = float(ofat_constant_parameters['er'])
    thickness_si = float(ofat_constant_parameters['thickness si (um)'])

    data_files = []
    for p in parameter_span:
        converged = False
        if parameter == 'sigma_s':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=p, zeta=span_df.loc['zeta']['base'],
                d_sf=span_df.loc['DSF']['base'], ef=span_df.loc['E']['base'], m=span_df.loc['m']['base'],
                h=span_df.loc['h']['base'], recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=span_df.loc['recovery electric field']['base']
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            converged = bool(simulation_parameters['converged'][0])
        if parameter == 'zeta':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=span_df.loc['sigma_s']['base'], zeta=p,
                d_sf=span_df.loc['DSF']['base'], ef=span_df.loc['E']['base'], m=span_df.loc['m']['base'],
                h=span_df.loc['h']['base'], recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=span_df.loc['recovery electric field']['base']
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            converged = bool(simulation_parameters['converged'][0])
        if parameter == 'DSF':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=span_df.loc['sigma_s']['base'], zeta=span_df.loc['zeta']['base'],
                d_sf=p, ef=span_df.loc['E']['base'], m=span_df.loc['m']['base'],
                h=span_df.loc['h']['base'], recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=span_df.loc['recovery electric field']['base']
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            converged = bool(simulation_parameters['converged'][0])
        if parameter == 'E':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=span_df.loc['sigma_s']['base'], zeta=span_df.loc['zeta']['base'],
                d_sf=span_df.loc['DSF']['base'], ef=p, m=span_df.loc['m']['base'],
                h=span_df.loc['h']['base'], recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=-p
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            if len(simulation_parameters) > 0:
                converged = bool(simulation_parameters['converged'][0])
            else:
                converged = False

        if parameter == 'm':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=span_df.loc['sigma_s']['base'], zeta=span_df.loc['zeta']['base'],
                d_sf=span_df.loc['DSF']['base'], ef=span_df.loc['E']['base'], m=p,
                h=span_df.loc['h']['base'], recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=span_df.loc['recovery electric field']['base']
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            converged = bool(simulation_parameters['converged'][0])

        if parameter == 'h':
            file_tag = pspan.create_filetag(
                time_s=time_s, temp_c=temp_c, sigma_s=span_df.loc['sigma_s']['base'], zeta=span_df.loc['zeta']['base'],
                d_sf=span_df.loc['DSF']['base'], ef=span_df.loc['E']['base'], m=span_df.loc['m']['base'],
                h=p, recovery_time=span_df.loc['recovery time']['base'],
                recovery_e_field=span_df.loc['recovery electric field']['base']
            )
            # Determine whether the simulation converged
            simulation_parameters = ofat_df[ofat_df['config file'] == file_tag + '.ini'].reset_index(drop=True)
            converged = bool(simulation_parameters['converged'][0])

        if converged:
            data_files.append({
                'parameter': parameter, 'value': p, 'pid_file': file_tag + '_simulated_pid.csv',
                'units': parameter_units[parameter]
            })

    n_files = len(data_files)
    c_map1 = mpl.cm.get_cmap('cool')
    normalize = mpl.colors.Normalize(vmin=0, vmax=n_files)
    plot_colors = [c_map1(normalize(i)) for i in range(n_files)]
    t_max = t_max_h * 3600.
    failure_times = np.empty(
        n_files, dtype=np.dtype([
            ('D (cm^2/s)', 'd'),
            ('t 5% loss (s)', 'd'),
        ])
    )

    mpl.rcParams.update(defaultPlotStyle)
    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    fig_p = plt.figure(1)
    fig_p.set_size_inches(4.75, 3.0, forward=True)
    # fig_p.subplots_adjust(hspace=0.0, wspace=0.0)
    # gs0_p = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_p, width_ratios=[1])
    # gs00_p = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0_p[0])
    # ax1_p = fig_p.add_subplot(gs00_p[0, 0])
    ax1_p = fig_p.add_subplot(1, 1, 1)

    fig_r = plt.figure(2)
    fig_r.set_size_inches(4.75, 3.0, forward=True)
    # fig_r.subplots_adjust(hspace=0.0, wspace=0.0)
    gs0_r = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_r, width_ratios=[1])
    gs00_r = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0_r[0])
    ax1_r = fig_r.add_subplot(1, 1, 1)
    pbar = trange(n_files, desc='Analyzing file', leave=True)
    for i, file_info in enumerate(data_files):
        # Read the simulated data from the csv file
        csv_file = os.path.join(base_path, 'batch_analysis', file_info['pid_file'])
        pid_df = pd.read_csv(csv_file)
        time_s = pid_df['time (s)'].to_numpy(dtype=float)
        power = pid_df['Pmpp (mW/cm^2)'].to_numpy(dtype=float)
        rsh = pid_df['Rsh (Ohm cm^2)'].to_numpy(dtype=float)
        time_h = time_s / 3600.

        t_interp = np.linspace(np.amin(time_s), np.amax(time_s), num=1000)
        f_p_interp = interpolate.interp1d(time_s, power, kind='linear')
        power_interp = f_p_interp(t_interp)
        idx_5 = (np.abs(power_interp / power_interp[0] - 0.95)).argmin()
        failure_times[i] = (file_info['value'], t_interp[idx_5].copy())
        sv_txt = r'{0} = ${1}$ $\mathregular{{{2}}}$'.format(
            file_info['parameter'], utils.latex_order_of_magnitude(file_info['value'], dollar=False),
            file_info['units']
        )

        ax1_p.plot(
            time_h, power / power[0], color=plot_colors[i], ls='-', label=sv_txt,
        )

        ax1_r.plot(
            time_h, rsh / rsh[0], color=plot_colors[i], ls='-', label=sv_txt,
        )
        pbar.set_description('Analyzing parameter {0}: {1}'.format(
            parameter, file_info['value'], file_info['units']
        ))
        pbar.update()
        pbar.refresh()

    ax1_p.set_ylabel('Normalized Power')
    ax1_r.set_ylabel('Normalized $R_{sh}$')

    ax1_p.set_xlim(0, t_max_h)
    ax1_r.set_xlim(0, t_max_h)
    ax1_p.set_xlabel('Time (hr)')
    ax1_r.set_xlabel('Time (hr)')

    # ax1.tick_params(labelbottom=True, top=False, right=True, which='both', labeltop=False)
    # ax2.tick_params(labelbottom=True, top=False, right=True, which='both')

    ax1_r.set_yscale('log')
    ax1_r.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=5))
    ax1_r.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=50, subs=np.arange(2, 10) * .1))

    ax1_p.xaxis.set_major_formatter(xfmt)
    ax1_p.xaxis.set_major_locator(mticker.MaxNLocator(7, prune=None))
    ax1_p.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1_p.yaxis.set_major_formatter(xfmt)
    ax1_p.yaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax1_p.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax1_r.xaxis.set_major_formatter(xfmt)
    ax1_r.xaxis.set_major_locator(mticker.MaxNLocator(7, prune=None))
    ax1_r.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # leg1 = ax1_p.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', borderaxespad=0., ncol=1, frameon=False)
    # leg2 = ax1_r.legend(bbox_to_anchor=(1.05, 1.), loc='upper left', borderaxespad=0., ncol=1, frameon=False)

    leg1 = ax1_p.legend(loc='upper right', frameon=False)
    leg2 = ax1_r.legend(loc='upper right', frameon=False)

    fig_p.tight_layout()
    fig_r.tight_layout()
    plt.show()

    output_file_tag = 'ofat_parameter_{}.svg'.format(slugify(value=parameter))
    fig_p.savefig(os.path.join(output_path, output_file_tag + '_power.png'), dpi=600)
    fig_p.savefig(os.path.join(output_path, output_file_tag + '_power.svg'), dpi=600)
    fig_r.savefig(os.path.join(output_path, output_file_tag + '_rsh.png'), dpi=600)
    fig_r.savefig(os.path.join(output_path, output_file_tag + '_rsh.svg'), dpi=600)

    df_degradation = pd.DataFrame(failure_times)
    df_degradation.to_csv(
        path_or_buf=os.path.join(
            output_path,
            output_file_tag + '.csv'
        ),
        index=False
    )
