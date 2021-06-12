#!/usr/bin/env python3
import pidsim.ml_pid as ml_pid
import numpy as np
import pandas as pd
import scipy.optimize as optimize
from scipy import interpolate
import os
import sys
from typing import List
import argparse
from pidsim.logwriter import LoggerWriter
import multiprocessing
import logging
from functools import partial
from datetime import datetime


def get_logger(output_path, file_tag_, **kwargs_):
    name = kwargs_.get('name', '')
    log_file = os.path.join(output_path, file_tag_ + "_{}.log".format(name))

    # logging.basicConfig(filename=logFile, level=logging.INFO)
    # get pnp_logger
    my_fit_logger = logging.getLogger('pid_fit_log')
    my_fit_logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    my_fit_logger.addHandler(fh)
    my_fit_logger.addHandler(ch)

    return my_fit_logger


def fobj(beta: List[float], **kwargs_) -> np.ndarray:
    temp = kwargs_.get('temp', 85)
    rate_source_ = kwargs_.get('rate_source', 1E-4)
    DSIN_ = kwargs_.get('DSIN', 3.92E-16)
    stress_voltage = kwargs_.get('stress_voltage', 3.75)
    L1_ = kwargs_.get('L1', 0.075)
    N1_ = int(kwargs_.get('N1', 100))
    tsteps_ = int(kwargs_.get('tsteps', 720))
    time_s = kwargs_.get('time_s', np.array([0]))
    rsh_norm_ = kwargs_.get('rsh_norm', np.array([0]))

    S0_ = 10 ** beta[0]
    h_ = 10 ** beta[1]
    DSF_ = 10 ** beta[2]
    print('new beta:')
    print(np.power(10, beta))
    model = ml_pid.simulate_rsh(
        S0=S0_, h=h_, DSF=DSF_, simulation_time=np.amax(time_s) * 1.1,
        temperature=temp, rate_source=rate_source_, DSIN=DSIN_,
        stress_voltage=stress_voltage, L1=L1_, m=1, time_steps=tsteps_,
        N1=N1_
    )

    f = interpolate.interp1d(model['time (s)'], model['rsh (ohms cm^2)'])
    y_pred = f(time_s)
    return np.log10(y_pred / y_pred[0]) - np.log10(rsh_norm_)


def func(beta, **kwargs_):
    temp = kwargs_.get('temp', 85)
    rate_source_ = kwargs_.get('rate_source', 1E-4)
    DSIN_ = kwargs_.get('DSIN', 3.92E-16)
    stress_voltage = kwargs_.get('stress_voltage', 3.75)
    L1_ = kwargs_.get('L1', 0.075)
    N1_ = int(kwargs_.get('N1', 100))
    tsteps_ = int(kwargs_.get('tsteps', 720))
    time_s = kwargs_.get('time_s', np.array([0]))

    S0_ = 10 ** beta[0]
    h_ = 10 ** beta[1]
    DSF_ = 10 ** beta[2]

    model = ml_pid.simulate_rsh(
        S0=S0_, h=h_, DSF=DSF_, simulation_time=np.amax(time_s) * 1.1,
        temperature=temp, rate_source=rate_source_, DSIN=DSIN_,
        stress_voltage=stress_voltage, L1=L1_, m=1, time_steps=tsteps_,
        N1=N1_
    )

    f = interpolate.interp1d(model['time (s)'], model['rsh (ohms cm^2)'])
    y_pred = f(time_s)

    return y_pred / y_pred[0]


def jac_pnp(beta, **kw):
    """Calculate derivatives for each parameter using pool."""
    temp = kw.get('temp', 85)
    rate_source_ = kw.get('rate_source', 1E-4)
    DSIN_ = kw.get('DSIN', 3.92E-16)
    stress_voltage = kw.get('stress_voltage', 3.75)
    L1_ = kw.get('L1', 0.075)
    N1_ = int(kw.get('N1', 100))
    tsteps_ = int(kw.get('tsteps', 720))
    time_s = kw.get('time_s', np.array([0]))
    rsh_norm_ = kw.get('rsh_norm', np.array([0]))
    print('Called jac_pnp')

    S0_ = 10 ** beta[0]
    h_ = 10 ** beta[1]
    DSF_ = 10 ** beta[2]
    # y0 = ml_pid.simulate_rsh(
    #     S0=S0_, h=h_, DSF=DSF_, simulation_time=np.amax(time_s) * 1.1,
    #     temperature=temp, rate_source=rate_source_, DSIN=DSIN_,
    #     stress_voltage=stress_voltage, L1=L1_, m=1, time_steps=tsteps_,
    #     N1=N1_
    # )

    EPS_ = np.finfo(np.float).eps
    delta = EPS_ ** (1 / 3)
    delta = 1E-1

    # forward
    # derivparams_forward = []
    derivparams = []
    for i in range(len(beta)):
        copy = np.array(beta)
        copy[i] += delta
        derivparams.append(copy)
    # backward
    # derivparams_backward = []
    for i in range(len(beta)):
        copy = np.array(beta)
        copy[i] -= delta
        derivparams.append(copy)

    # results_forward = pool.map(partial(func, **kw), derivparams_forward)
    # results_backward = pool.map(partial(func, **kw), derivparams_backward)
    results = np.array(pool.map(partial(func, **kw), derivparams))
    [m, n] = results.shape
    idx = int(m / 2)
    results_forward = results[0:idx, :]
    results_backward = results[idx::,:]
    derivs = [(rf - rb) / (2.0 * delta) for rf, rb in zip(results_forward, results_backward)]
    return np.array(derivs).T


if __name__ == '__main__':
    # Read the csv file from the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data', required=True, type=str, help='The csv data file (.csv)'
    )
    parser.add_argument(
        '-T', '--temp', required=False, type=float, help='The temperature in celsius'
    )

    parser.add_argument(
        '-M', '--tsteps', required=False, type=int, help='The number of time steps'
    )

    parser.add_argument(
        '-r', '--rate', required=False, type=float, help='The rate of Na ingress at the source'
    )
    parser.add_argument(
        '-dsin', '--dsin', required=False, type=float, help='The Na diffusivity in SiNx'
    )
    parser.add_argument(
        '-v', '--voltage', required=True, type=float, help='The stress voltage'
    )
    parser.add_argument(
        '-L', '--L', required=False, type=float, help='The thickness of SiNx'
    )
    parser.add_argument(
        '-N', '--N', required=False, type=int, help='The number of points in SiNx'
    )

    args = parser.parse_args()
    csv_file = args.data

    if args.tsteps is not None:
        tsteps = args.tsteps
    else:
        tsteps = 720
    if args.temp is not None:
        temperature = args.temp
    else:
        temperature = 85
    if args.rate is not None:
        rate_source = args.rate
    else:
        rate_source = 1E-4
    if args.dsin is not None:
        DSIN = args.dsin
    else:
        DSIN = 3.92E-16
    if args.L is not None:
        L1 = args.L
    else:
        L1 = 0.075
    if args.N is not None:
        N1 = args.N
    else:
        N1 = 100

    pool = multiprocessing.Pool(10)

    e_field = args.voltage / L1 / 100
    out_path = os.path.dirname(csv_file)
    file_tag = os.path.basename(csv_file)
    file_tag = os.path.splitext(file_tag)[0]
    file_tag = '{0}_{1:.2E}Vpcm'.format(file_tag, e_field)
    now = datetime.now()
    time_stamp = now.strftime('%Y%m%d-%H%M%S')
    file_tag = '{0}_{1}'.format(file_tag, time_stamp)

    exp_df = pd.read_csv(csv_file)
    time = np.array(exp_df['time (s)'].values)
    rsh = np.array(exp_df['Rsh (ohm cm^2)'].values)
    rsh_norm = rsh / rsh[0]
    exp_data = np.empty(len(rsh), dtype=np.dtype([('time (s)', 'd'), ('rsh (ohms cm^2)', 'd')]))
    for i, t, r in zip(range(len(time)), time, rsh_norm):
        exp_data[i] = (t, r)

    kwargs = dict(
        temperature=temperature,
        tsteps=tsteps,
        rate_source=rate_source,
        DSIN=DSIN,
        er=7.0,
        stress_voltage=args.voltage,
        L1=L1,
        N1=N1,
        time_s=time,
        rsh_norm=rsh_norm
    )

    all_tol = 1E-6  # np.finfo(np.float64).eps
    fit_logger = get_logger(output_path=out_path, file_tag_=file_tag, name='fit')
    fit_logger.info(exp_data.tostring())

    stdout_ = sys.stdout  # Keep track of the previous value.
    stderr_ = sys.stderr

    sys.stdout = LoggerWriter(fit_logger.info)
    sys.stderr = LoggerWriter(fit_logger.warning)

    EPS = np.finfo(np.float64).eps

    beta_0 = [10, -12, -19]

    res = optimize.least_squares(
        fobj, beta_0,
        kwargs=kwargs,
        bounds=([8, -14, -21], [13, -8, -13]),
        xtol=all_tol,
        ftol=all_tol,
        gtol=EPS,
        verbose=2,
        jac=jac_pnp,  # '2-point',
        # tr_options={'regularize': True},
        x_scale='jac',
        loss='linear',
        # diff_step=EPS ** 0.3,
        # max_nfev=100
    )

    sys.stdout = stdout_
    sys.stderr = stderr_

    popt_log = res.x
    popt = np.power(10, popt_log)

    prediction = ml_pid.simulate_rsh(
        S0=popt[0], h=popt[1], DSF=popt[2], simulation_time=time.max(),
        temperature=temperature, rate_source=rate_source, DSIN=DSIN,
        stress_voltage=args.voltage, L1=L1, m=1.0, time_steps=tsteps
    )

    pred_df = pd.DataFrame(data=prediction)
    out_csv = os.path.join(out_path, file_tag + '_fit.csv')
    pred_df.to_csv(path_or_buf=out_csv, index=False)
    popt_ds = {
        'S0 (1/cm^2)': [popt[0]],
        'h (cm/s)': [popt[1]],
        'DSF (cm^2/s)': [popt[2]]
    }
    popt_df = pd.DataFrame(data=popt_ds)
    fit_logger.info(popt_df.to_string())
    out_csv = os.path.join(out_path, file_tag + '_popt.csv')
    popt_df.to_csv(path_or_buf=out_csv, index=False)


