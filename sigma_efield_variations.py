"""
This program will create all the necessary input files to run pnp transport simulations with combinatorial variations
on the values of the surface concentration and electric field.

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
path_to_output = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\sigma_efield_asu_20200820'
# Simulation time in h
simulation_time_h = 96
# Temperature in °C
temperature_c = 85
# Relative permittivity of SiNx
er = 7.0
# Thickness of the SiNx um
thickness_sin = 80E-3
# Modeled thickness of Si um
thickness_si = 1.0
# Number of time steps
t_steps = 720
# Number of elements in the sin layer
x_points_sin = 100
# number of elements in the Si layer
x_points_si = 200
# Background concentration in cm^-3
cb = 1E-20
# The rate of ingress at the surface (1/s)
zeta = 1E-3
# The surface mass transfer coefficient at the SiNx/Si interface in cm/s
h = 1E-10
# The segregation coefficient at the SiNx/Si interface
segregation_coefficient = 1.
# The diffusion coefficient of Na in the stacking fault (cm^2/s)
d_sf = 1E-14

e_fields = np.array([1E-2, 1E-1, 1.])
sigmas = np.array([1E10, 1E11, 1E12])

if __name__ == '__main__':
    pspan.sigma_efield_variations(
        sigmas=sigmas, efields=e_fields, out_dir=path_to_output, zeta=zeta, simulation_time=simulation_time_h * 3600.,
        dsf=d_sf, h=h, m=segregation_coefficient, temperature_c=temperature_c, er=7.0, thickness_sin=thickness_sin,
        thickness_si=thickness_si, t_steps=t_steps, x_points_sin=x_points_sin, base_concentration=cb
    )
