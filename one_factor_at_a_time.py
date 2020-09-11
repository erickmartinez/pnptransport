"""
This program will create all the necessary input files to run pnp transport simulations with 'one factor at a time'
variations.

The variations on the relevant parameters are described in  pidsim.parameter_span.one_factor_at_a_time documentation.
These variations are submitted through a csv data file.

The rest of the parameters are assumed to be constant over all the simulations.

Besides the input files, the code will generate a database as a csv file with all the simulations to be run and the
parameters used for each simulation.

@author: <erickrmartinez@gmail.com>
"""
import numpy as np
import pidsim.parameter_span as pspan

# The path to the csv file with the conditions of the different variations
csv_file = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\one_factor_at_a_time_lower_20200828_h=1E-12.csv'
# Simulation time in h
simulation_time_h = 96.
# Temperature in °C
temperature_c = 85
# Relative permittivity of SiNx
er = 7.0
# Thickness of the SiNx um
thickness_sin = 75E-3
# Modeled thickness of Si um
thickness_si = 1.0
# Number of time steps
t_steps = 720
# Number of elements in the sin layer
x_points_sin = 100
# number of elements in the Si layer
x_points_si = 100
# Background concentration in cm^-3
cb = 1E-20


if __name__ == '__main__':
    pspan.one_factor_at_a_time(
        csv_file=csv_file, simulation_time=simulation_time_h*3600, temperature_c=temperature_c, er=er,
        thickness_sin=thickness_sin, thickness_si=thickness_si, t_steps=t_steps, x_points_sin=x_points_sin,
        x_points_si=x_points_si, base_concentration=cb
    )
