import pidsim.ml_simulator as ml_kinetics
import pnptransport.finitesource as pnpfs
import logging
import os
import numpy as np
from datetime import datetime
from pidsim.parameter_span import create_filetag
import traceback
import random


def get_logger(output_path, file_tag, **kwargs):
    name = kwargs.get('name', '')
    log_file = os.path.join(output_path, file_tag + "_{}.log".format(name))

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


def simulate_rsh(S0: float, h: float, DSF: float, *args, **kwargs) -> np.ndarray:
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
    base_path = kwargs.get('base_path', r'/home/fenics/shared/fenics/shared/pid_fit/temp')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    now = datetime.now()
    time_stamp = now.strftime('%Y%m%d-%H%M%S_%f')
    out_path = os.path.join(base_path, time_stamp+'_{0}'.format(random.randint(0,10)))

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
    x1points = int(kwargs.get('N1', 50))
    # The thickness of the simulated Si layer in um
    L2 = float(kwargs.get('L2', 1.0))
    # The number of points in the layer
    x2points = int(kwargs.get('N2', 50))
    # The segregation coefficient at the SiNx/Si interface
    m = float(kwargs.get('m', 1.0))

    e_field = voltage / L1 / 100

    # The configuration file
    # Logging
    # The base filename for the output
    file_tag = create_filetag(
        time_s=t_max,
        temp_c=temp_c,
        sigma_s=S0,
        zeta=rate_source,
        d_sf=D2cms,
        ef=e_field,
        m=m,
        h=h
    )

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    try:
        h5_file = os.path.join(out_path, file_tag + ".h5")
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
            h5_storage=h5_file,
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

    rfr_simulator = ml_kinetics.MLSim(h5_transport_file=h5_file)
    time_s = rfr_simulator.time_s
    requested_indices = rfr_simulator.get_requested_time_indices(time_s)
    rsh = rfr_simulator.rsh_time_series(requested_indices=requested_indices)

    result = np.empty(len(rsh), dtype=np.dtype([('time (s)', 'd'), ('rsh (ohms cm^2)', 'd')]))
    for i, t, r in zip(range(len(time_s)), time_s, rsh):
        result[i] = (t, r)
    print(result)

    return result

