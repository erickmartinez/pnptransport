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
    # Number of moles of sodium to introduce in the source
    surface_concentration = config.getfloat(section='global', option='surface_concentration', fallback=1E11)
    # The rate of ingress of ions at the source
    zeta = config.getfloat(section='global', option='zeta', fallback=1E-5)

    # The diffusion coefficient of layer 1 in cm2/s
    D1cms = config.getfloat(section='sinx', option='d')
    er = config.getfloat(section='global', option='er')

    # The electric fields
    voltage = config.getfloat(section='sinx', option='stress_voltage')  # volts
    # The geometry
    L1 = config.getfloat(section='sinx', option='thickness')  # um
    # The number of points in the sinx layer
    x1points = config.getint(section='sinx', option='npoints')

    # Estimate the diffusion coefficients for the given temperature
    TempK = temp_c + 273.15

    try:
        rpath = os.path.join(out_path, 'single_layer')
        if not os.path.exists(rpath):
            os.makedirs(rpath)
        h5FileName = os.path.join(rpath, file_tag + ".h5")
        myLogger = getLogger(rpath, file_tag, name='SL')
        vfb, t_sim, x1i, c1i, p1i, c_max = pnpfs.single_layers_zero_flux(
            D1cms=D1cms,
            thickness_sinx=L1,
            tempC=temp_c,
            voltage=voltage,
            time_s=t_max,
            surface_concentration=surface_concentration,
            fcallLogger=myLogger,
            xpoints_sinx=x1points,
            tsteps=tsteps,
            h5_storage=h5FileName,
            zeta=zeta,
            er=7.0,
            z=1.0,
            maxr_calls=5,
            debug=True
        )

    except Exception as e:
        traceback.print_exc()
        print('Error occured trying to simulate.')
        print(e)
