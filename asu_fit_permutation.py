import numpy as np
import pandas as pd
from scipy import interpolate
import pnptransport.finitesource as pnpfs
import os
import multiprocessing
import logging
from functools import partial
from datetime import datetime
import pidsim.ml_simulator as ml_kinetics
from pidsim.parameter_span import create_filetag
import traceback
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import h5py

csv_file = r'/home'

S0_s = [1E8, 1E9, 1E10, 1E11] # 1/cm^2
h_s = [1E-8, 1E-10, 1E-12, 1E-14] # cm/s
DSF_s = [1E-14, 1E-16, 1E-18, 1E-20] # cm^2/s

tsteps = 360
temperature = 85
rate_source = 1E-4
DSIN = 3.92E-16
L1 = 0.075
N1 = 50
L2 = 1
N2 = 50

voltage = 0.375
e_field = voltage / L1 / 100
csv_file = r'/home/fenics/shared/fenics/shared/pid_fit/PID_mc_BSF_4_ready.csv'


def cost_function(y, rsh_norm_) -> float:
    # x = model['time (s)']
    # y = model['rsh (ohms cm^2)']
    # f = interpolate.interp1d(x, y)
    # y_pred = f(time_s)
    y = np.array(y)
    m = len(y)
    diff = np.log10(y / y[0]) - np.log10(rsh_norm_)
    r = 0.5 * np.dot(diff.T, diff) / m
    return r


def func(beta, **kwargs_):
    temp = kwargs_.get('temp', 85)
    rate_source_ = kwargs_.get('rate_source', 1E-4)
    DSIN_ = kwargs_.get('DSIN', 3.92E-16)
    stress_voltage = kwargs_.get('stress_voltage', 3.75)
    L1_ = kwargs_.get('L1', 0.075)
    N1_ = int(kwargs_.get('N1', 100))
    tsteps_ = int(kwargs_.get('tsteps', 360))
    time_s = kwargs_.get('time_s', np.array([0]))
    # h5_file = kwargs_.get('h5file', None)

    S0_ = float(beta[0])
    h_ = float(beta[1])
    DSF_ = float(beta[2])
    h5_file = beta[3]
    print('func, h5file: {0}'.format(h5_file))

    kw = dict(
        simulation_time=np.amax(time_s) * 1.1,
        temperature=temp, rate_source=rate_source_, DSIN=DSIN_,
        stress_voltage=stress_voltage, L1=L1_, m=1, time_steps=tsteps_,
        N1=N1_, h5file=h5_file
    )

    model = simulate_rsh(
        S0=S0_, h=h_, DSF=DSF_, **kw
    )

    f = interpolate.interp1d(model['time (s)'], model['rsh (ohms cm^2)'])
    y_pred = f(time_s)

    base_dir = os.path.dirname(h5_file)
    base_name = os.path.splitext(os.path.basename(h5_file))[0]
    csv_file = os.path.join(base_dir, base_name + '.csv')
    csv_df = pd.DataFrame(
        data={
            'time (s)': time_s,
            'Rsh (ohm cm^2)': y_pred,
            'Rsh norm': y_pred / y_pred[0]
        }
    )
    csv_df.to_csv(path_or_buf=csv_file, index=False)

    return y_pred


def get_logger(output_path, file_tag, name):
    log_file_tag = '{0}_{1}.log'.format(file_tag, name)
    log_file = os.path.join(output_path, log_file_tag)

    # logging.basicConfig(filename=logFile, level=logging.INFO)
    # get pnp_logger
    fit_logger = logging.getLogger('simlog')
    fit_logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    #    formatter 	= logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #    ch.setFormatter(formatter)
    #    fh.setFormatter(formatter)

    # add the handlers to the logger
    fit_logger.addHandler(fh)
    fit_logger.addHandler(ch)

    return fit_logger


def simulate_rsh(S0: float, h: float, DSF: float, h5file: float, *args, **kwargs) -> np.ndarray:
    """
    This function simulates the PID kinetics for a p-Si Al-BSF PV module using a trained RFR model.
    Na transport is simulated using FEniCS.

    Parameters
    ----------
    S0: float
        The initial Na surface concentration at the source in 1/cm^2
    h: float
        The surface mass transfer coefficient at the SiNx/Si interface
    DSF: float
        The diffusion coefficient of Na in the stacking fault
    args: list
        Positional arguments
    kwargs: dict
        Keyword arguments

    Returns
    -------
    np.ndarray:
        Rsh as a function of PID stress time
    """
    # The total simulation time in seconds
    t_max = float(kwargs.get('simulation_time', 345600))
    # Temperature in celsius
    temp_c = float(kwargs.get('temperature', 85))
    # The bulk concentration (cm-3)
    c_bulk = float(kwargs.get('cb', 1E-20))
    # The number of time steps
    tsteps = int(kwargs.get('time_steps', 720))
    # The surface concentration of the sourceW
    surface_concentration = S0
    # The rate of ingress at the source in 1/s.
    rate_source = float(kwargs.get('rate_source', 1E-4))

    # The diffusion coefficient of layer 1 in cm2/s
    D1cms = float(kwargs.get('DSIN', 3.92E-16))
    # The diffusion coefficient of layer 1 in cm2/s
    D2cms = DSF
    # The dielectric constant of the dielectric
    er = float(kwargs.get('er', 7.0))
    # The electric fields
    voltage = float(kwargs.get('stress_voltage', 3.75))  # volts
    # The geometry
    L1 = float(kwargs.get('L1', 0.075))  # um
    # The number of points in the sinx layer
    x1points = int(kwargs.get('N1', 100))
    # The thickness of the simulated Si layer in um
    L2 = float(kwargs.get('L2', 1.0))
    # The number of points in the layer
    x2points = int(kwargs.get('N2', 100))
    # The segregation coefficient at the SiNx/Si interface
    m = float(kwargs.get('m', 1.0))

    e_field = voltage / L1 / 100

    # The configuration file
    # Logging

    try:
        out_path = os.path.dirname(h5file)
        file_tag = os.path.splitext(os.path.basename(h5file))[0]

        myLogger = get_logger(out_path, file_tag, name='pnp')
        _, _, _, _, _, _, _, _ = pnpfs.two_layers_constant_flux(
            D1cms=D1cms, D2cms=D2cms,
            h=h, m=m,
            thickness_sinx=L1,
            thickness_si=L2,
            tempC=temp_c,
            voltage=voltage,
            time_s=t_max,
            surface_concentration=surface_concentration,
            rate=rate_source,
            recovery_time_s=0,
            recovery_voltage=0,
            fcallLogger=myLogger,
            xpoints_sinx=x1points,
            xpoints_si=x2points,
            tsteps=tsteps,
            h5_storage=h5file,
            er=er,
            z=1.0,
            maxr_calls=2,
            trapping=False,
            c_fp=0.0,
            cbulk=c_bulk,
            debug=True
        )

    except Exception as e:
        traceback.print_exc()
        print('Error occured trying to simulate.')
        print(e)

    rfr_simulator = ml_kinetics.MLSim(h5_transport_file=h5file)
    time_s = rfr_simulator.time_s
    requested_indices = rfr_simulator.get_requested_time_indices(time_s)
    rsh = rfr_simulator.rsh_time_series(requested_indices=requested_indices)

    result = np.empty(len(rsh), dtype=np.dtype([('time (s)', 'd'), ('rsh (ohms cm^2)', 'd')]))
    for i, t, r in zip(range(len(time_s)), time_s, rsh):
        result[i] = (t, r)
    print(result)

    return result


if __name__ == "__main__":
    root_dir = os.path.dirname(csv_file)
    file_tag = os.path.basename(csv_file)
    file_tag = os.path.splitext(file_tag)[0]

    exp_df = pd.read_csv(csv_file)
    exp_df = exp_df[exp_df['time (s)'] <= 345600]
    time = np.array(exp_df['time (s)'].values)
    rsh = np.array(exp_df['Rsh (ohm cm^2)'].values)
    rsh_norm = rsh / rsh[0]

    base_path = r'/home/fenics/shared/fenics/shared/pid_fit/combinations'

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    now = datetime.now()
    time_stamp = now.strftime('%Y%m%d-%H%M%S_%f')
    out_path = os.path.join(base_path, time_stamp)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # The total simulation time in seconds
    t_max = time.max()
    # Temperature in celsius
    temp_c = 85
    # The bulk concentration (cm-3)
    c_bulk = 1E-20
    # The number of time steps
    tsteps = 360

    # The rate of ingress at the source in 1/s.
    rate_source = 1E-4

    # The diffusion coefficient of layer 1 in cm2/s
    D1cms = 3.92E-16

    # The dielectric constant of the dielectric
    er = 7.0
    e_field = voltage / L1 / 100

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    kw = dict(
        temperature=temperature,
        tsteps=tsteps,
        rate_source=rate_source,
        DSIN=DSIN,
        er=7.0,
        stress_voltage=voltage,
        L1=L1,
        N1=N1,
        L2=L1,
        N2=N2,
        time_s=time,
        rsh_norm=rsh_norm,
        out_path=out_path
    )

    now = datetime.now()
    time_stamp = now.strftime('%Y%m%d-%H%M%S')
    file_tag = '{0}_{1}'.format(file_tag, time_stamp)

    n_simulations = len(S0_s) * len(h_s) * len(DSF_s)

    pool = multiprocessing.Pool(70)

    sim_params = []

    for s in S0_s:
        for hi in h_s:
            for d in DSF_s:
                filetag = create_filetag(
                    time_s=t_max,
                    temp_c=temp_c,
                    sigma_s=s,
                    zeta=rate_source,
                    d_sf=d,
                    ef=e_field,
                    m=1.0,
                    h=hi
                )
                h5file = os.path.join(out_path, filetag + ".h5")
                params = np.array([s, hi, d, h5file])
                sim_params.append(params)

    results = pool.map(partial(func, **kw), sim_params)
    permutations_df = pd.DataFrame(data=sim_params)
    permutations_df.columns = [
        'S0 (1/cm^2)', 'h (cm/s)', 'D_SF (cm^2/s)', 'h5 file'
    ]
    pool.close()
    costs = np.empty(n_simulations, dtype=np.float)

    for i, r in permutations_df.iterrows():
        csv_file = os.path.splitext(r['h5 file'])[0] + '.csv'
        df = pd.read_csv(csv_file)
        y = df['Rsh norm'].values
        costs[i] = cost_function(y=y, rsh_norm_=rsh_norm)

    permutations_df['cost'] = costs
    csv_grid_file = os.path.join(root_dir, file_tag + ".h5")
    permutations_df.to_csv(path_or_buf=csv_grid_file + '_grid.csv', index=False)

    idx_min = np.argmin(costs)
    opt_h5 = permutations_df.iloc[idx_min]['h5 file']
    rpath = os.path.dirname(opt_h5)
    opt_basename = os.path.splitext(os.path.basename(opt_h5))[0]
    opt_csv = os.path.join(rpath, opt_basename + '.csv')
    opt_df = pd.read_csv(opt_csv)

    # Load my style
    with open('plotstyle.json', 'r') as style_file:
        mpl.rcParams.update(json.load(style_file)['defaultPlotStyle'])

    # Plot PID data
    fig_pid = plt.figure(1)
    fig_pid.set_size_inches(4.0, 3.0, forward=True)
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
        time / 3600, opt_df['Rsh norm'].values,  ls='-', color='tab:red',
        label='Best fit'
    )

    xfmt = ScalarFormatter(useMathText=True)
    xfmt.set_powerlimits((-3, 3))

    ax_pid.xaxis.set_major_formatter(xfmt)
    ax_pid.xaxis.set_major_locator(mticker.MaxNLocator(12, prune=None))
    ax_pid.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    ax_pid.set_yscale('log')

    leg = ax_pid.legend(loc='best', frameon=True)
    opt_fig_tag = os.path.join(root_dir, file_tag + '_optimized')

    fig_pid.tight_layout()

    fig_pid.savefig(opt_fig_tag + '.png', dpi=300)
    fig_pid.savefig(opt_fig_tag + '.svg', dpi=600)

    fig_pid.show()
