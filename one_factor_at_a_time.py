import numpy as np
import pidsim.parameter_span as pspan

csv_file = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\one_factor_at_a_time_lower_MLS.csv'
# Simulation time in h
simulation_time_h = 96
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
x_points_si = 200
# Background concentration in cm^-3
cb = 1E-20


if __name__ == '__main__':
    pspan.one_factor_at_a_time(
        csv_file=csv_file, simulation_time=simulation_time_h*3600, temperature_c=temperature_c, er=er,
        thickness_sin=thickness_sin, thickness_si=thickness_si, t_steps=t_steps, x_points_sin=x_points_sin,
        x_points_si=x_points_si, base_concentration=cb
    )
