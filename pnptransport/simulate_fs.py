#!/usr/bin/env python3
import numpy as np
import configparser
import os
import argparse
import logging
import pnptransport.finitesource as pnpfs
import traceback


def getLogger(out_path, filetag, **kwargs):
    name = kwargs.get('name', '')
    logFile = os.path.join(out_path, filetag + "_{}.log".format(name))

    # logging.basicConfig(filename=logFile, level=logging.INFO)
    # get the myLogger
    pnp_logger = logging.getLogger('simlog')
    pnp_logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    #    formatter 	= logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    #    ch.setFormatter(formatter)
    #    fh.setFormatter(formatter)

    # add the handlers to the logger
    pnp_logger.addHandler(fh)
    pnp_logger.addHandler(ch)

    return pnp_logger


if __name__ == '__main__':
    import platform

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', required=True, type=str, help='The configuration file (.ini)'
    )

    args = parser.parse_args()
    config_file = args.config

    if platform == 'Windows':
        config_file = r'\\\?\\' + config_file

    if not os.path.exists(config_file):
        raise Exception('Could not find file \'{}\''.format(config_file))

    # Get the current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    # The full path to the configuration file
    configPath = os.path.join(cwd, config_file)
    # Instantiate the config parser
    config = configparser.ConfigParser()
    # Load the configuration file
    config.read(config_file)

    # The configuration file
    # Logging
    # The base filename for the output
    file_tag = config.get(section='global', option='filetag')
    logFile = file_tag + ".log"

    full_path = os.path.abspath(config_file)
    base_path = os.path.dirname(os.path.dirname(config_file))

    out_path = os.path.join(base_path, 'results')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # The total simulation time in seconds
    t_max = config.getfloat(section='global', option='simulation_time')
    # Temperature in celsius
    temp_c = config.getfloat(section='global', option='temperature')
    # The bulk concentration (cm-3)
    c_bulk = config.getfloat(section='global', option='cb')
    # The number of time steps
    tsteps = config.getint(section='global', option='time_steps')
    # RECOVERY SIMULATIONS
    # Recovery time in seconds (defaults to 0)
    recovery_time = config.getfloat(section='global', option='recovery_time', fallback=0)
    # Recovery voltage in volts (defaults to 0)
    recovery_voltage = config.getfloat(section='global', option='recovery_voltage', fallback=0)
    # Number of moles of sodium to introduce in the source
    surface_concentration = config.getfloat(section='global', option='surface_concentration', fallback=1E11)
    # Surface mass transfer coefficient at the source in cm/s. (Defaults to 1E-10 cm/s)
    zeta = config.getfloat(section='global', option='zeta', fallback=1E-4)
    trap_corrected = config.getboolean(section='global', option='trapping', fallback=False)
    c_fp = config.getfloat(section='global', option='c_fp', fallback=1E-3)

    # The diffusion coefficient of layer 1 in cm2/s
    D1cms = config.getfloat(section='sinx', option='d')
    # The diffusion coefficient of layer 1 in cm2/s
    D2cms = config.getfloat(section='si', option='d')
    # The dielectric constant of the dielectric
    er = config.getfloat(section='global', option='er')

    # The electric fields
    voltage = config.getfloat(section='sinx', option='stress_voltage')  # volts
    # The geometry
    L1 = config.getfloat(section='sinx', option='thickness')  # um
    # The number of points in the sinx layer
    x1points = config.getint(section='sinx', option='npoints')
    # The thickness of the simulated Si layer in um
    L2 = config.getfloat(section='si', option='thickness')
    # The number of points in the layer
    x2points = config.getint(section='si', option='npoints')
    # The surface mass transfer coefficient at the SiNx/Si interface in cm/s
    h = config.getfloat(section='global', option='h')
    # The segregation coefficient at the SiNx/Si interface
    m = config.getfloat(section='global', option='m')

    # Estimate the diffusion coefficients for the given temperature
    TempK = temp_c + 273.15

    try:
        rpath = os.path.join(out_path, 'constant_flux')
        if not os.path.exists(rpath):
            os.makedirs(rpath)
        h5FileName = os.path.join(rpath, file_tag + ".h5")
        myLogger = getLogger(rpath, file_tag, name='pnp')
        vfb, tsim, x1sim, c1sim, psim, x2sim, c2sim, cmax = pnpfs.two_layers_constant_flux(
            D1cms=D1cms, D2cms=D2cms,
            h=h, m=m,
            thickness_sinx=L1,
            thickness_si=L2,
            tempC=temp_c,
            voltage=voltage,
            time_s=t_max,
            surface_concentration=surface_concentration, h_surface=zeta,
            recovery_time_s=recovery_time,
            recovery_voltage=recovery_voltage,
            fcallLogger=myLogger,
            xpoints_sinx=x1points,
            xpoints_si=x2points,
            tsteps=tsteps,
            h5_storage=h5FileName,
            er=7.0,
            z=1.0,
            maxr_calls=5,
            trapping=trap_corrected,
            c_fp=c_fp,
            debug=True
        )

    except Exception as e:
        traceback.print_exc()
        print('Error occured trying to simulate.')
        print(e)
