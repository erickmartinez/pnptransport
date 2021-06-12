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
import os

output_folder = r'G:\Shared drives\FenningLab2\Projects\PVRD1\Simulations\Sentaurus PID\results\analysis\finite_source'
data_root = r'G:\Shared drives\FenningLab2\Projects\PVRD1\Simulations\Sentaurus PID\results\3D'
input_folder = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\images\sentaurus_dsf'
ml_simulations_csv = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20201028\ofat_db.csv'

sentaurus_db_csv = r'sentaurus_pid_dsf.csv'
literature_db_csv = r'literature_pid_dsf.csv'
batch_analysis_folder = 'batch_analysis_rfr_t'


conditions = {
    'efield': 1E4,
    'S': 1E10,
    'k': 1E-4,
    'h': 1E-12
}

if __name__ == '__main__':
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_files_df = pd.read_csv(os.path.join(input_folder, sentaurus_db_csv))
    literature_files_df = pd.read_csv(os.path.join(input_folder, literature_db_csv))
    ofat_df = pd.read_csv(os.path.join(ml_simulations_csv))

    # ML dataroot
    ml_data_root = os.path.join(os.path.dirname(ml_simulations_csv), batch_analysis_folder)
    print(ml_data_root)

    # print(ofat_df.columns)
    with open('plotstyle.json', 'r') as f:
        mpl.rcParams.update(json.load(f)['defaultPlotStyle'])

    # Define the color palette
    cm = cmap.get_cmap('rainbow_r')

    nplots = len(input_files_df)  # + len(literature_files_df)
    normalize = mpl.colors.Normalize(vmin=0, vmax=(nplots-1))
    plot_colors = [cm(normalize(i)) for i in range(nplots)]
    plot_marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p', '*'))

    fig = plt.figure(1)
    fig.set_size_inches(9.5, 3.5, forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.35)
    gs0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig, width_ratios=[1])
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0[0])
    ax = fig.add_subplot(gs00[0, 0])

    # Plot Sentaurus results
    for i, r in input_files_df.iterrows():
        dsf = float(r['DSF'])
        lbl = r'$\mathregular{{D_{{SF}}}} = {0}$'.format(utils.latex_order_of_magnitude(dsf))
        efficiency_csv = os.path.join(
            data_root, r['folder'], 'analysis_plots', 'rsh_data.csv'
        )
        efficiency_df: pd.DataFrame = pd.read_csv(efficiency_csv)

        power = np.array(efficiency_df['Rsh (Ohms cm2)'])
        p0 = power[0]
        idx = np.round(power, decimals=0) < np.round(p0, decimals=0)
        time_h = np.array(efficiency_df['time (h)'].values)  # / 3600.

        if len(power) > 0:
            ax.plot(
                time_h, power / power[0], color=plot_colors[i],  # marker=next(plot_marker),
                label=lbl, ls='-', fillstyle='none', zorder=i+len(literature_files_df)
            )

    cm = cmap.get_cmap('Greys_r')

    nplots = len(literature_files_df)  # + len(literature_files_df)
    normalize = mpl.colors.Normalize(vmin=0, vmax=(nplots+1))
    plot_colors = [cm(normalize(i)) for i in range(nplots)]
    plot_marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p', '*'))

    for i, lf in literature_files_df.iterrows():
        fn = lf['file']
        time_units = lf['time_units']
        label = lf['label']
        if lf['type'] == 'power':
            column_names = ['time', 'power']
        else:
            column_names = ['time', 'Rsh']
        lit_df = pd.read_csv(fn, skiprows=0, header=0, names=column_names, index_col=False)

        if time_units == 'min':
            time_lf = lit_df['time'].to_numpy() / 60
        elif time_units == 's':
            time_lf = lit_df['time'].to_numpy() / 3600
        elif time_units == 'h':
            time_lf = lit_df['time'].to_numpy()
        else:  # Assume hours
            time_lf = lit_df['time'].to_numpy()

        # t_max = max(time_lf.max(), t_max)

        if lf['type'] == 'Rsh':
            rsh = lit_df['Rsh']
            if not lf['normalized']:
                rsh = rsh / rsh[0]
            ax.plot(
                time_lf, rsh, fillstyle='none',
                label=label, color=plot_colors[i],
                marker=next(plot_marker), zorder=i,
                ls='--'
            )

    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('Normalized $R_{sh}$')
    ax.set_xlim(0, 96.)
    # ax.set_ylim(0.8, 1.05)

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(12, prune=None))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax.set_yscale('log')

    # ax.yaxis.set_major_formatter(xfmt)
    # ax.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    # ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    leg = ax.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left', borderaxespad=0., ncol=2)

    # Plot Sentaurus vs ML time series
    cm = cmap.get_cmap('rainbow_r')
    nplots = len(input_files_df)  # + len(literature_files_df)
    normalize = mpl.colors.Normalize(vmin=0, vmax=(nplots-1))
    plot_colors = [cm(normalize(i)) for i in range(nplots)]
    plot_marker = itertools.cycle(('o', 's', '^', 'v', '>', '<', 'd', 'p', '*'))

    fig_ml = plt.figure(2)
    fig_ml.set_size_inches(8.5, 3.5, forward=True)
    fig_ml.subplots_adjust(hspace=0.1, wspace=0.35)
    gs0_ml = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_ml, width_ratios=[1])
    gs00_ml = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs0_ml[0])
    ax_ml = fig_ml.add_subplot(gs00_ml[0, 0])

    for i, r in input_files_df.iterrows():
        dsf = float(r['DSF'])
        lbl = r'$\mathregular{{D_{{SF}}}} = {0}$'.format(utils.latex_order_of_magnitude(dsf))
        efficiency_csv = os.path.join(
            data_root, r['folder'], 'analysis_plots', 'rsh_data.csv'
        )
        efficiency_df: pd.DataFrame = pd.read_csv(efficiency_csv)

        power = np.array(efficiency_df['Rsh (Ohms cm2)'])
        p0 = power[0]
        idx = np.round(power, decimals=0) < np.round(p0, decimals=0)
        time_h = np.array(efficiency_df['time (h)'].values) #/ 3600.


        # Find the corresponding ML time series
        q = '`sigma_s (cm^-2)` == {0} & '.format(conditions['S'])
        q += '`zeta (1/s)` ==  {0} & '.format(conditions['k'])
        q += '`D_SF (cm^2/s)` == {0} & '.format(dsf)
        q += '`E (V/cm)` == {0} & '.format(conditions['efield'])
        q += '`h (cm/s)` == {0}'.format(conditions['h'])
        ml_dsf_df = ofat_df.query(q).reset_index().head(1)
        ml_pid_csv = ml_dsf_df['config file'].values
        if len(ml_pid_csv) > 0:
            file_tag = os.path.splitext(ml_pid_csv[0])[0] + '_simulated_pid.csv'
            path_to_ml_csv = os.path.join(ml_data_root, file_tag)
            if os.path.exists(path_to_ml_csv):
                print(path_to_ml_csv)
                ml_simulated_pid_df = pd.read_csv(path_to_ml_csv)
                time_ml = np.array(ml_simulated_pid_df['time (s)']) / 3600
                mpp_ml = np.array(ml_simulated_pid_df['Rsh (Ohm cm^2)'])
                lbl = r'$\mathregular{{D_{{SF}}}} = {0}$, ML'.format(utils.latex_order_of_magnitude(dsf))
                ax_ml.plot(
                    time_ml, mpp_ml / mpp_ml[0], color=plot_colors[i],  #marker=next(plot_marker),
                    label=lbl, ls='--', fillstyle='none', zorder=i
                )
        if len(power) > 0:
            ax_ml.plot(
                time_h, power / power[0], color=plot_colors[i],  # marker=next(plot_marker),
                label=lbl, ls='-', fillstyle='none', zorder=i+nplots
            )

        # print(file_tag)

    ax_ml.set_xlabel('Time (hr)')
    ax_ml.set_ylabel('Normalized $R_{sh}$')
    ax_ml.set_xlim(0, 96.)
    # ax_ml.set_ylim(0.8, 1.05)
    
    ax_ml.set_yscale('log')

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    ax_ml.xaxis.set_major_formatter(xfmt)
    ax_ml.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
    ax_ml.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # ax_ml.yaxis.set_major_formatter(xfmt)
    # ax_ml.yaxis.set_major_locator(mticker.MaxNLocator(5, prune=None))
    # ax_ml.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    leg2 = ax_ml.legend(bbox_to_anchor=(1.1, 1.0), loc='upper left', borderaxespad=0., ncol=2)

    fig.tight_layout()
    fig_ml.tight_layout()
    fig.savefig(os.path.join(output_folder, 'failure_time_dsf_rsh.svg'), dpi=600)
    fig.savefig(os.path.join(output_folder, 'failure_time_dsf_rsh.png'), dpi=600)

    fig_ml.savefig(os.path.join(output_folder, 'failure_time_dsf_ml_rsh.svg'), dpi=600)
    fig_ml.savefig(os.path.join(output_folder, 'failure_time_dsf_ml_rsh.png'), dpi=600)
    plt.show()
