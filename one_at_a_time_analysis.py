"""
This code runs basic analysis on simulations that were computed using the 'one at a time analysis'.

You must provide the path to the csv database with the parameters of each simulation.

Functionality:

1. Plot the last concentration profile over the layer stack.
2. Plot the Rsh(t) estimated with the series resistor model.
3. Estimate the integrated sodium concentration in SiNx and Si at the end of the simulation.
"""
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import pidsim.rsh as prsh
import pidsim.pmpp_rf as pmpp_rf
import h5py
import os
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy import integrate
import pnptransport.utils as utils
from tqdm import tqdm

path_to_csv = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200813\ofat_db.csv'
path_to_results = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200813\results'

color_map = 'Blues'

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
        path_to_csv = r'\\?\\' + path_to_csv
        path_to_results = r'\\?\\' + path_to_results

    # Create an analysis folder within the base dir for the database file
    working_path = os.path.dirname(path_to_csv)
    analysis_path = os.path.join(working_path, 'batch_analysis')
    # If the folder does not exists, create it
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    # Read the database of simulations
    simulations_df = pd.read_csv(filepath_or_buffer=path_to_csv)
    # pick only the simulations that converged
    simulations_df = simulations_df[simulations_df['converged'] == 1].reset_index(drop=True)
    # Count the simulations
    n_simulations = len(simulations_df)
    integrated_final_concentrations = np.empty(n_simulations, dtype=np.dtype([
        ('C_SiNx average final (atoms/cm^3)', 'd'), ('C_Si average final (atoms/cm^3)', 'd')
    ]))
    # Load the style
    mpl.rcParams.update(defaultPlotStyle)
    # Get the color map
    cm = mpl.cm.get_cmap(color_map)
    # Show at least the first 6 figures
    max_displayed_figures = 6
    fig_counter = 0
    for i, r in simulations_df.iterrows():
        filetag = os.path.splitext(r['config file'])[0]
        simga_s = r['sigma_s (cm^-2)']
        zeta = r['zeta (1/s)']
        dsf = r['D_SF (cm^2/s)']
        e_field = r['E (V/cm)']
        h = r['h (cm/s)']
        m = r['m']
        time_max = r['time (s)']
        temp_c = r['temp (C)']

        source_str1 = r'$\sigma_{{\mathrm{{s}}}} = {0} \; (\mathrm{{cm^{{-2}}}})$'.format(
            utils.latex_order_of_magnitude(simga_s))
        source_str2 = r'$\zeta = {0} \; (\mathrm{{1/s}})$'.format(utils.latex_order_of_magnitude(zeta))
        e_field_str = r'$E = {0} \; (\mathrm{{V/cm}})$'.format(utils.latex_order_of_magnitude(e_field))
        h_str = r'$h = {0} \; (\mathrm{{cm/s}})$'.format(utils.latex_order_of_magnitude(h))
        temp_str = r'${0:.0f} \; (\mathrm{{°C}})$'.format(temp_c)
        dsf_str = r'$D_{{\mathrm{{SF}}}} = {0} \; (\mathrm{{cm^2/s}})$'.format(utils.latex_order_of_magnitude(dsf))
        # Normalize the time scale
        normalize = mpl.colors.Normalize(vmin=1E-3, vmax=(time_max / 3600.))
        # Get a 20 time points geometrically spaced
        requested_time = utils.geometric_series_spaced(max_val=time_max, min_delta=600, steps=20)
        # Get the full path to the h5 file
        path_to_h5 = os.path.join(path_to_results, filetag + '.h5')
        # Create the concentration figure
        fig_c = plt.figure()
        fig_c.set_size_inches(5.0, 3.0, forward=True)
        fig_c.subplots_adjust(hspace=0.1, wspace=0.1)
        gs_c_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_c)
        # 1 column for the concentration profile in SiNx
        # 1 column for the concentration profile in Si
        # 1 column for the colorbar
        gs_c_00 = gridspec.GridSpecFromSubplotSpec(
            nrows=1, ncols=2, subplot_spec=gs_c_0[0], wspace=0.0, hspace=0.1, width_ratios=[2.5, 3]
        )
        ax_c_0 = fig_c.add_subplot(gs_c_00[0, 0])
        ax_c_1 = fig_c.add_subplot(gs_c_00[0, 1])

        # Axis labels
        ax_c_0.set_xlabel(r'Depth (nm)')
        ax_c_0.set_ylabel(r'[Na] ($\mathregular{cm^{-3}}$)')
        # Title to the sinx axis
        ax_c_0.set_title(r'${0}\; \mathrm{{V/cm}}, {1:.0f}\; \mathrm{{°C}}$'.format(
            utils.latex_order_of_magnitude(e_field), temp_c
        ))
        # Set the ticks for the Si concentration profile axis to the right
        ax_c_1.yaxis.set_ticks_position('right')
        # Title to the si axis
        ax_c_1.set_title(r'$D_{{\mathrm{{SF}}}} = {0}\; \mathrm{{cm^2/s}},\; E=0$'.format(
            utils.latex_order_of_magnitude(dsf)
        ))
        ax_c_1.set_xlabel(r'Depth (um)')
        # Log plot in the y axis
        ax_c_0.set_yscale('log')
        ax_c_1.set_yscale('log')
        ax_c_0.set_ylim(bottom=1E10, top=1E20)
        ax_c_1.set_ylim(bottom=1E10, top=1E20)
        # Set the ticks for the SiNx log axis
        ax_c_0.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
        ax_c_0.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=60, subs=np.arange(2, 10) * .1))
        # Set the ticks for the Si log axis
        ax_c_1.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
        ax_c_1.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=60, subs=np.arange(2, 10) * .1))
        ax_c_1.tick_params(axis='y', left=False, labelright=False)
        # Configure the ticks for the x axis
        ax_c_0.xaxis.set_major_locator(mticker.MaxNLocator(4, prune=None))
        ax_c_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        ax_c_1.xaxis.set_major_locator(mticker.MaxNLocator(3, prune='lower'))
        ax_c_1.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
        # Change the background colors
        # ax_c_0.set_facecolor((0.89, 0.75, 1.0))
        # ax_c_1.set_facecolor((0.82, 0.83, 1.0))
        # Create the integrated concentration figure
        fig_s = plt.figure()
        fig_s.set_size_inches(4.75, 3.0, forward=True)
        fig_s.subplots_adjust(hspace=0.1, wspace=0.1)
        gs_s_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_s)
        gs_s_00 = gridspec.GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_s_0[0], hspace=0.1,
        )
        ax_s_0 = fig_s.add_subplot(gs_s_00[0, 0])
        # Set the axis labels
        ax_s_0.set_xlabel(r'Time (h)')
        ax_s_0.set_ylabel(r'$\bar{C}$ ($\mathregular{cm^{-3}}$)')
        # Set the limits for the x axis
        ax_s_0.set_xlim(left=0, right=time_max / 3600.)
        # Make the y axis log
        ax_s_0.set_yscale('log')
        # Set the ticks for the y axis
        ax_s_0.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, numticks=6))
        ax_s_0.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, numticks=60, subs=np.arange(2, 10) * .1))
        # Set the ticks for the x axis
        # Configure the ticks for the x axis
        ax_s_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
        ax_s_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        # Create the rsh figure
        fig_r = plt.figure()
        fig_r.set_size_inches(4.75, 3.0, forward=True)
        fig_r.subplots_adjust(hspace=0.1, wspace=0.1)
        gs_r_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig_r)
        gs_r_00 = gridspec.GridSpecFromSubplotSpec(
            nrows=1, ncols=1, subplot_spec=gs_r_0[0], hspace=0.1,
        )
        ax_r_0 = fig_r.add_subplot(gs_r_00[0, 0])
        # Set the axis labels
        ax_r_0.set_xlabel(r'Time (h)')
        ax_r_0.set_ylabel(r'$R_{\mathrm{sh}}$ ($\mathrm{\Omega\cdot cm^2}}$)')

        with h5py.File(path_to_h5, 'r') as hf:
            # Get the time dataset
            time_s = np.array(hf['time'])
            # Get the sinx group
            grp_sinx = hf['L1']
            # get the si group
            grp_si = hf['L2']
            # Get the position vector in SiNx in nm
            x_sin = np.array(grp_sinx['x']) * 1000.
            thickness_sin = np.max(x_sin)
            x_si = np.array(grp_si['x']) - thickness_sin / 1000.
            x_sin = x_sin - thickness_sin
            thickness_si = np.amax(x_si)
            n_profiles = len(time_s)
            requested_indices = utils.get_indices_at_values(x=time_s, requested_values=requested_time)
            time_profile = np.empty(len(requested_indices))

            model_colors = [cm(normalize(t)) for t in time_s / 3600.]
            scalar_maps = mpl.cm.ScalarMappable(cmap=cm, norm=normalize)
            with tqdm(requested_indices, leave=True, position=0) as pbar:
                for j, idx in enumerate(requested_indices):
                    time_j = time_s[idx] / 3600.
                    time_profile[j] = time_j
                    # Get the specific profile
                    ct_ds = 'ct_{0:d}'.format(idx)
                    c_sin = np.array(grp_sinx['concentration'][ct_ds])
                    c_si = np.array(grp_si['concentration'][ct_ds])
                    color_j = cm(normalize(time_j))
                    ax_c_0.plot(x_sin, c_sin, color=color_j, zorder=0)
                    ax_c_1.plot(x_si, c_si, color=color_j, zorder=0)
                    pbar.set_description('Extracting profile {0} at time {1:.1f} h...'.format(ct_ds, time_j))
                    pbar.update()
                    pbar.refresh()

            # Estimate the integrated concentrations as a function of time for each layer
            c_sin_int = np.empty(n_profiles)
            c_si_int = np.empty(n_profiles)
            with tqdm(range(n_profiles), leave=True, position=0) as pbar:
                for j in range(n_profiles):
                    # Get the specific profile
                    ct_ds = 'ct_{0:d}'.format(j)
                    c_sin = np.array(grp_sinx['concentration'][ct_ds])
                    c_si = np.array(grp_si['concentration'][ct_ds])
                    c_sin_int[j] = abs(integrate.simps(c_sin, -x_sin )) / thickness_sin
                    c_si_int[j] = abs(integrate.simps(c_si, x_si)) / thickness_si
                    pbar.set_description('Integrating profile at time {0:.1f} h: S_N: {1:.2E}, S_S: {2:.3E} cm^-2'.format(
                        time_s[j] / 3600.,
                        c_sin_int[j],
                        c_si_int[j]
                    ))
                    pbar.update()
                    pbar.refresh()

        ax_s_0.plot(time_s / 3600., c_sin_int, label=r'$\mathregular{SiN_x}$')
        ax_s_0.plot(time_s / 3600., c_si_int, label=r'Si')
        # ax_s_0.plot(time_s / 3600., c_si_int + c_sin_int, label=r'Si + $\mathregular{SiN_x}$')

        integrated_final_concentrations[i] = (c_sin_int[-1], c_si_int[-1])

        leg = ax_s_0.legend(loc='lower right', frameon=True)

        # Set the limits for the x axis of the concentration plot
        ax_c_0.set_xlim(left=np.amin(x_sin), right=np.amax(x_sin))
        ax_c_1.set_xlim(left=np.amin(x_si), right=np.amax(x_si))
        # Add the color bar
        divider = make_axes_locatable(ax_c_1)
        cax = divider.append_axes("right", size="7.5%", pad=0.03)
        cbar = fig_c.colorbar(scalar_maps, cax=cax)
        cbar.set_label('Time (h)\n', rotation=90, fontsize=14)
        cbar.ax.tick_params(labelsize=11)

        plot_c_sin_txt = source_str1 + '\n' + source_str2
        ax_c_0.text(
            0.95, 0.95,
            plot_c_sin_txt,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax_c_0.transAxes,
            fontsize=11,
            color='k'
        )

        plot_c_si_txt = h_str + '\n$m=1$'
        ax_c_1.text(
            0.95, 0.95,
            plot_c_si_txt,
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax_c_1.transAxes,
            fontsize=11,
            color='k'
        )

        # Identify layers
        ax_c_0.text(
            0.05, 0.015,
            r'$\mathregular{SiN_x}$',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax_c_0.transAxes,
            fontsize=11,
            fontweight='bold',
            color='k'
        )

        ax_c_1.text(
            0.05, 0.015,
            'Si',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax_c_1.transAxes,
            fontsize=11,
            fontweight='bold',
            color='k'
        )

        # set the y axis limits for the integrated concentration plot
        ax_s_0.set_ylim(bottom=1E5, top=1E20)
        title_str = source_str1 + ', ' + source_str2 + ', ' + dsf_str

        plot_txt = e_field_str + '\n' + temp_str + '\n' + h_str
        ax_s_0.set_title(title_str)
        ax_s_0.text(
            0.65, 0.95,
            plot_txt,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax_s_0.transAxes,
            fontsize=11,
            color='k'
        )

        # rsh_analysis = prsh.Rsh(h5_transport_file=path_to_h5)
        pmpp_analysis = pmpp_rf.Pmpp(h5_transport_file=path_to_h5)
        time_s = pmpp_analysis.time_s
        time_h = time_s / 3600.
        requested_indices = pmpp_analysis.get_requested_time_indices(time_s)
        pmpp = pmpp_analysis.pmpp_time_series(requested_indices=requested_indices)

        ax_r_0.plot(time_h, pmpp / pmpp[0])
        ax_r_0.set_xlim(0, np.amax(time_h))
        # ax_r_0.set_yscale('log')
        ax_r_0.set_xlabel('time (h)')
        # ax_r_0.set_ylabel('$R_{\mathrm{sh}}\;(\Omega \cdot \mathregular{cm^2})$')
        ax_r_0.set_ylabel('Normalized Power')

        ax_r_0.xaxis.set_major_locator(mticker.MaxNLocator(6, prune=None))
        ax_r_0.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        title_str = source_str1 + ', ' + source_str2 + ', ' + dsf_str

        plot_txt = e_field_str + '\n' + temp_str + '\n' + h_str
        ax_r_0.set_title(title_str)
        ax_r_0.text(
            0.65, 0.95,
            plot_txt,
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax_r_0.transAxes,
            fontsize=11,
            color='k'
        )

        fig_c.tight_layout()
        fig_s.tight_layout()
        fig_r.tight_layout()

        fig_c.savefig(os.path.join(analysis_path, filetag + '_c.png'), dpi=600)
        fig_s.savefig(os.path.join(analysis_path, filetag + '_s.png'), dpi=600)
        fig_r.savefig(os.path.join(analysis_path, filetag + '_p.png'), dpi=600)

        plt.close(fig_c)
        plt.close(fig_s)
        plt.close(fig_r)

        del fig_c, fig_s, fig_r

    simulations_df['C_SiNx average final (atoms/cm^3)'] = integrated_final_concentrations['C_SiNx average final (atoms/cm^3)']
    simulations_df['C_Si average final (atoms/cm^3)'] = integrated_final_concentrations['C_Si average final (atoms/cm^3)']

    simulations_df.to_csv(os.path.join(analysis_path, 'ofat_analysis.csv'), index=False)


