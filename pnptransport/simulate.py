#!/usr/bin/env python3
import numpy as np
import configparser
import os
import argparse
import logging
import pnptransport.constantsource as pnpcs
import pnptransport.limtedsource as pnpsl
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
    # Deprectaing model selection
    parser.add_argument(
        '-m', '--model', required=True, choices=['erf', 'np', 'pnp'],
        help='The model to use for the simulation: erf, np or pnp.'
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
    config.read(configPath)
    # Get the model string
    model = args.model

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
    # True if a limited source (defaults to False)
    source_limited = config.getboolean(section='global', option='source_limited', fallback=False)
    # Number of moles of sodium to introduce in the source
    source_na_moles = config.getfloat(section='global', option='source_na_moles', fallback=1)
    # Surface mass transfer coefficient at the source in cm/s. (Defaults to 1E-10 cm/s)
    source_h = config.getfloat(section='global', option='source_h', fallback=1E-10)
    # The surface concentration at the source in atoms/cm^2 (defaults to 1E11)
    source_max_surface_concentration = config.getfloat(
        section='global',
        option='source_max_surface_concentration',
        fallback=1E11
    )

    # The diffusion coefficient of layer 1 in cm2/s
    D1cms = config.getfloat(section='sinx', option='d')
    # The diffusion coefficient of layer 1 in cm2/s
    D2cms = config.getfloat(section='si', option='d')
    # The dielectric constant of the dielectric
    er = config.getfloat(section='global', option='er')
    # The concentration at the source
    Cs = config.getfloat(section='global', option='cs')

    # The electric fields
    voltage = config.getfloat(section='sinx', option='stress_voltage')  # volts
    # The geometry
    L1 = config.getfloat(section='sinx', option='thcikness')  # um
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
        if model == 'pnp':
            rpath = os.path.join(out_path, 'pnp')
            if not os.path.exists(rpath):
                os.makedirs(rpath)
            h5FileName = os.path.join(rpath, file_tag + "_pnp.h5")
            myLogger = getLogger(rpath, file_tag, name='pnp')
            if source_limited:
                vfb, tsim, x1sim, c1sim, psim, x2sim, c2sim, cmax = pnpsl.two_layers_source_lim(
                    D1cms=D1cms, D2cms=D2cms,
                    Cs=Cs,
                    h=h, m=m,
                    thickness_sinx=L1,
                    thickness_si=L2,
                    tempC=temp_c,
                    voltage=voltage,
                    time_s=t_max,
                    na_moles=source_na_moles, h0=source_h,
                    max_surface_concentration=source_max_surface_concentration,
                    recovery_time_s=recovery_time,
                    fcallLogger=myLogger,
                    xpoints_sinx=x1points,
                    xpoints_si=x2points,
                    tsteps=tsteps,
                    h5_storage=h5FileName,
                    er=7.0,
                    z=1.0,
                    maxr_calls=5,
                    debug=True
                )
            else:
                vfb, tsim, x1sim, c1sim, psim, x2sim, c2sim, cmax = pnpcs.two_layers_constant_source(
                    D1cms=D1cms, D2cms=D2cms,
                    Cs=Cs,
                    h=h, m=m,
                    thickness_sinx=L1,
                    thickness_si=L2,
                    tempC=temp_c,
                    voltage=voltage,
                    time_s=t_max,
                    recovery_time_s=recovery_time,
                    fcallLogger=myLogger,
                    xpoints_sinx=x1points,
                    xpoints_si=x2points,
                    tsteps=tsteps,
                    h5_storage=h5FileName,
                    er=7.0,
                    z=1.0,
                    maxr_calls=5,
                    debug=True
                )

    except Exception as e:
        traceback.print_exc()
        print('Error occured trying to simulate.')
        print(e)
