import numpy as np
import pnptransport.utils as utils
import pandas as pd
import platform
from string import Template
from datetime import datetime
import os

template_file_fs = 'input_template_finite_source_1.txt'
# The D0 for sinx
d0_sinx = 1E-14
# The activation energy for sinx
ea_sinx = 0.1

sim_db_dtype = np.dtype([
    ('config file', 'U200'), ('sigma_s (cm^-2)', 'd'), ('zeta (1/s)', 'd'), ('D_SF (cm^2/s)', 'd'),
    ('E (V/cm)', 'd'), ('h (cm/s)', 'd'), ('m', 'd'), ('time (s)', 'd'), ('temp (C)', 'd'), ('bias (V)', 'd'),
    ('recovery time (s)', 'd'), ('recovery E (V/cm)', 'd'), ('recovery  bias (V)', 'd'),
    ('thickness sin (um)', 'd'), ('thickness si (um)', 'd'), ('er', 'd'), ('cb (cm^-3)', 'd'), ('t_steps', 'i'),
    ('x_points sin', 'i'), ('x_points si', 'i')
])


def one_factor_at_a_time(csv_file: str, simulation_time: float, temperature_c: float, er: float = 7.0,
                         thickness_sin: float = 75E-3, thickness_si: float = 1, t_steps: int = 720,
                         x_points_sin: int = 100, x_points_si: int = 200, base_concentration: float = 1E-20, ):
    """
    Generates input files and batch script to run one-factor-at-a-time parameter variation

    Parameters
    ----------
    csv_file: str
        The path to the csv file containing the base case and the parameter scans to simulate:
        Format of the file

            +----------------+-----------+-----------------+
            | Parameter name | Base case | span            |
            +================+===========+=================+
            | sigma_s        | 1E+11     | 1E10,1E11,...   |
            +----------------+-----------+-----------------+
            | zeta           | 1E-4      | 1E-4,1E-3,...   |
            +----------------+-----------+-----------------+
            | DSF            | 1E-14     | 1E-12,1E-14,... |
            +----------------+-----------+-----------------+
            | E              | 1E4       | 1E2,1E4,...     |
            +----------------+-----------+-----------------+
            | m              | 1         | 1               |
            +----------------+-----------+-----------------+
            | h              | 1E-8      | 1E-8,1E-7,...   |
            +----------------+-----------+-----------------+
    simulation_time: float
        The total simulation time in s.
    temperature_c: float
        The simulation temperature in °C
    er: float
        The relative permittivity of SiNx. Default 7.0
    thickness_sin: float
        The thickness of the SiNx layer in um. Default: 0.075
    thickness_si: float
        The thickness of the Si layer in um. Default 1 um
    t_steps: int
        The number of time steps for the integration.
    x_points_sin: int
        The number of grid points in the SiN layer
    x_points_si: int
        The number of grid points in the Si layer.
    base_concentration: float
        The background impurity concentration in cm^-3. Default 1E-20 cm\ :sup:`-3` \.
    """
    # Read the csv
    ofat_df = pd.read_csv(filepath_or_buffer=csv_file, index_col=0)
    # Extract the values of the parameters
    # The surface concentration in atoms/cm^2
    sigma_s = ofat_df.loc['sigma_s']
    # The rate of ingress in 1/s
    zeta = ofat_df.loc['zeta']
    # The diffusion coefficient of Na in the SF (cm^2/s)
    dsf = ofat_df.loc['DSF']
    # The electric field in the SiNx layer (V/cm)
    e_field = ofat_df.loc['E']
    # The segregation coeffiecient at the SiNx/Si interface
    segregation_coefficient = ofat_df.loc['m']
    # The surface mass transfer coefficient at the SiNx/Si interface (cm/s(
    h = ofat_df.loc['h']
    # Check if recovery is specified in the dataset, otherwise assume no-recovery
    recovery_e_field = {'base': 0.0, 'span': 0.0}
    recovery_time = {'base': 0.0, 'span': 0.0}
    if 'recovery time' in ofat_df.index:
        recovery_time = ofat_df.loc['recovery time']
        if 'recovery electric field' in ofat_df.index:
            recovery_e_field = ofat_df.loc['recovery electric field']
        else:
            recovery_e_field = {'base': -e_field['base'], 'span': -e_field['base']}

    # Define the base case
    base_case = {
        'sigma_s': sigma_s['base'],
        'zeta': zeta['base'],
        'dsf': dsf['base'],
        'e_field': e_field['base'],
        'segregation_coefficient': segregation_coefficient['base'],
        'h': h['base'],
        'recovery_time': recovery_time['base'],
        'recovery_e_field': recovery_e_field['base']
    }


    # Define the parameter spans
    span_sigma_s = string_list_to_float(sigma_s['span'])
    span_zeta = string_list_to_float(zeta['span'])
    span_dsf = string_list_to_float(dsf['span'])
    span_e_field = string_list_to_float(e_field['span'])
    span_segregation_coefficient = string_list_to_float(segregation_coefficient['span'])
    span_h = string_list_to_float(h['span'])
    span_recovery_time = np.array([0.0])
    span_recovery_e_field = np.array([0.0])
    if 'recovery time' in ofat_df.index:
        span_recovery_time = string_list_to_float(recovery_time['span'])
        if 'recovery electric field' in ofat_df.index:
            span_recovery_e_field = string_list_to_float(recovery_e_field['span'])
        else:
            span_recovery_e_field = np.array([-e_field['base']])

    # count sigma_s simulations
    n_sigma_s = len(span_sigma_s) - 1
    n_zeta = len(span_zeta) - 1
    n_dsf = len(span_dsf) - 1
    n_e_field = len(span_e_field) - 1
    n_segregation_coefficient = len(span_segregation_coefficient) - 1
    n_h = len(span_h) - 1
    n_recovery_time = len(span_recovery_time) - 1
    n_recovery_e_field = len(span_recovery_e_field) - 1

    # How many simulations in total?
    total_simulations = n_sigma_s + n_zeta + n_dsf + n_e_field + n_segregation_coefficient + n_h + 1

    # Create a data structure to index all the simulations
    ofat_simulations_db = np.empty(total_simulations, dtype=sim_db_dtype)

    out_dir = os.path.join(os.path.dirname(csv_file), 'inputs')
    if platform.system() == 'Windows':
        out_dir = r'\\?\\' + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    config_filename = create_input_file(
        simulation_time=simulation_time, temperature_c=temperature_c, sigma_s=sigma_s['base'], zeta=zeta['base'],
        d_sf=dsf['base'], e_field=e_field['base'], segregation_coefficient=segregation_coefficient['base'],
        h=h['base'], thickness_sin=thickness_sin, thickness_si=thickness_si, base_concentration=base_concentration,
        er=er, t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
        recovery_time=recovery_time['base'], recovery_e_field=recovery_e_field['base']
    )

    bias = sin_bias_from_e(e_field=e_field['base'], thickness_sin=thickness_sin)
    bias_recovery = sin_bias_from_e(e_field=recovery_e_field['base'], thickness_sin=thickness_sin)
    ofat_simulations_db[0] = (
        config_filename, sigma_s['base'], zeta['base'], dsf['base'], e_field['base'], h['base'],
        segregation_coefficient['base'], simulation_time, temperature_c, bias,
        recovery_time['base'], recovery_e_field['base'], bias_recovery, thickness_sin, thickness_si,
        er, base_concentration, t_steps, x_points_sin, x_points_si

    )
    print('Created base case:')

    i = 1
    # Generate inputs for sigma_s span
    for v in span_sigma_s:
        if v != base_case['sigma_s']:
            params = base_case.copy()
            params['sigma_s'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery =  sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created sigma_s span {0:.1E} atoms/s'.format(v))
            i += 1

    for v in span_zeta:
        if v != base_case['zeta']:
            params = base_case.copy()
            params['zeta'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created zeta span {0:.1E} ML/s'.format(v))
            i += 1

    for v in span_dsf:
        if v != base_case['dsf']:
            params = base_case.copy()
            params['dsf'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created D_SF span {0:.1E} cm^2/s'.format(v))
            i += 1

    for v in span_e_field:
        if v != base_case['e_field']:
            params = base_case.copy()
            params['e_field'] = v
            params['recovery_e_field'] = -v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created E span {0:.1E} V/cm'.format(v))
            i += 1

    for v in span_h:
        if v != base_case['h']:
            params = base_case.copy()
            params['h'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created h span {0:.1E} cm/s'.format(v))
            i += 1

    for v in span_segregation_coefficient:
        if v != base_case['segregation_coefficient']:
            params = base_case.copy()
            params['segregation_coefficient'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created m span {0:.1E}'.format(v))
            i += 1

    for v in span_recovery_time:
        if v != base_case['recovery_time']:
            params = base_case.copy()
            params['recovery_time'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created recovery time span {0:.1E}'.format(v))
            i += 1

    for v in span_recovery_e_field:
        if v != base_case['recovery_e_field']:
            params = base_case.copy()
            params['recovery_e_field'] = v
            config_filename = create_span_file(
                param_list=params, simulation_time=simulation_time,
                temperature_c=temperature_c, thickness_sin=thickness_sin, thickness_si=thickness_si, er=er,
                t_steps=t_steps, x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir,
                base_concentration=base_concentration
            )

            bias = sin_bias_from_e(e_field=params['e_field'], thickness_sin=thickness_sin)
            bias_recovery = sin_bias_from_e(e_field=params['recovery_e_field'], thickness_sin=thickness_sin)
            ofat_simulations_db[i] = (
                config_filename, params['sigma_s'], params['zeta'], params['dsf'], params['e_field'], params['h'],
                params['segregation_coefficient'], simulation_time, temperature_c, bias,
                params['recovery_time'], params['recovery_e_field'], bias_recovery, thickness_sin, thickness_si,
                er, base_concentration, t_steps, x_points_sin, x_points_si

            )
            print('Created recovery E span {0:.1E}'.format(v))
            i += 1

    simulations_df = pd.DataFrame(ofat_simulations_db)
    simulations_df.to_csv(path_or_buf=os.path.join(out_dir, 'ofat_db.csv'), index=False)


def string_list_to_float(the_list: str) -> np.ndarray:
    """
    Takes a string containing a comma-separated list and converts it to a numpy array of floats
    
    Parameters
    ----------
    the_list: str
        The comma-separated list

    Returns
    -------
    np.ndarray:
        The corresponding array
    """
    return np.array(the_list.split(',')).astype(float)


def create_filetag(time_s: float, temp_c: float, sigma_s: float, zeta: float, d_sf: float, ef: float, m: float,
                   h: float, recovery_time: float = 0, recovery_e_field: float = 0) -> str:
    """
    Create the file_tag for the simulation input file

    Parameters
    ----------
    time_s: float
        The simulation time in seconds.
    temp_c: float
        The temperature in °C
    sigma_s: float
        The surface concentration of the source, in atoms/ cm\ :sup:`2` \.
    zeta: float
        The rate of ingress in 1/s
    d_sf: float
        The diffusivity at the SF in cm\ :sup:`2` \/s
    ef: float
        The applied electric field in SiNx in V/cm
    m: float
        The segregation coefficient
    h: float
        The surface mass transfer coefficient at the SiNx/Si interface in cm/s
    recovery_time: float
        The simulated recovery time (additional to the PID simulations) in s. Default: 0.
    recovery_e_field: float
        The electric field applied under recovery (ideally with sign opposite to the PID stress) units: V.
        Default: 0 V

    Returns
    -------
    str:
        The file_tag
    """
    filetag = 'constant_source_flux_{0:.0f}'.format(time_s / 3600)
    filetag += '_{0:.0f}C'.format(temp_c)
    filetag += '_{0:.0E}pcm2'.format(sigma_s)
    filetag += '_z{0:.0E}ps'.format(zeta)
    filetag += '_DSF{0:.0E}'.format(d_sf)
    filetag += '_{0:.0E}Vcm'.format(ef)
    filetag += '_h{0:.0E}'.format(h)
    filetag += '_m{0:.0E}'.format(m)
    if recovery_time > 0:
        filetag += '_rt{0:.0f}h'.format(recovery_time / 3600.)
        filetag += '_rv{0:.0E}Vcm'.format(recovery_e_field)
    return filetag


def sin_bias_from_e(e_field: float, thickness_sin: float) -> float:
    """
    Estimates the bias in SiNx based on the value of the electric field and the thickness of the layer.

    Parameters
    ----------
    e_field: float
        The electric field in the SiNx layer (V/cm)
    thickness_sin: float
        The thickness of the SiNx layer in (um).

    Returns
    -------
    float:
        The corresponding bias in V
    """
    return e_field * thickness_sin * 1E-4


def create_span_file(param_list: dict, simulation_time: float, temperature_c: float,
                     thickness_sin: float, thickness_si: float, base_concentration: float, er: float, t_steps: int,
                     x_points_sin: int, x_points_si: int, out_dir: str):
    """
    A wrapper for create_input_file that unpacks the values of the parameter span contained in param_list

    Parameters
    ----------
    param_list: dict
        A dictionary with the values of the parameters that are being varied. Must contain:
        - sigma_s: The surface concentration in atoms/cm\ :sup:`2` \
        - zeta: the rate of transfer in 1/s
        - dsf: the diffusion coefficient of Na in the SF
        - e_field: The electric field in V/cm
        - segregation coefficient: The segregation coefficient at the SiNx/Si interface
        - h: The surface mass transfer coefficient at the SiNx/Si interface
        - recovery_time: The recovery time added to the simulation
        - recovery_e_field: The electric field applied under recovery
    simulation_time: float
        The simulation time in seconds
    temperature_c: float
        The simulation temperature in °C
    thickness_sin: float
        The thickness of the SiNx layer in um.
    thickness_si: float
        The thickness of the simulated SF in um
    base_concentration: float
        The bulk base impurity concentration for all layers in 1/cm\ :sup:`3` \
    er: float
        The relative permittivity of SiNx
    t_steps: int
        The number of time steps to simulate.
    x_points_sin: int
        The number of elements to simulate in the SiNx layer.
    x_points_si: int
        The number of elements to simulate in the Si layer.
    out_dir: str
        The path to the output directory


    Returns
    -------
    str:
        The name of the input file for the simulation.
    """
    config_filename = create_input_file(
        simulation_time=simulation_time, temperature_c=temperature_c, sigma_s=param_list['sigma_s'],
        zeta=param_list['zeta'], d_sf=param_list['dsf'], e_field=param_list['e_field'],
        segregation_coefficient=param_list['segregation_coefficient'], h=param_list['h'], thickness_sin=thickness_sin,
        thickness_si=thickness_si, base_concentration=base_concentration, er=er, t_steps=t_steps,
        x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir, recovery_time=param_list['recovery_time'],
        recovery_e_field=param_list['recovery_e_field']
    )
    return config_filename


def create_input_file(simulation_time: float, temperature_c: float, sigma_s: float, zeta: float, d_sf: float,
                      e_field: float, segregation_coefficient: float, h: float, thickness_sin: float,
                      thickness_si: float, base_concentration: float, er: float, t_steps: int,
                      x_points_sin: int, x_points_si: int, out_dir: str, recovery_time: float = 0.,
                      recovery_e_field: float = 0) -> str:
    """
    Creates an inputfile for the finite source simulation

    Parameters
    ----------
    simulation_time: float
        The simulation time in seconds
    temperature_c: float
        The simulation temperature °C
    sigma_s: float
        The surface concentration of the source in atoms/cm\ :sup:`2` \
    zeta: float
        The surface rate of ingress of Na in (1/s)
    d_sf: float
        The diffusion coefficient in the SF in cm\ :sup:`2` \/s
    e_field: float
        The electric field in SiNx in V/cm
    segregation_coefficient: float
        The segregation coefficient at the SiNx/Si interface
    h: float
        The surface mass transfer coefficient at the SiNx/Si interface in cm/s
    thickness_sin: float
        The thickness of the SiNx layer in um.
    thickness_si: float
        The thickness of the Si layer in um
    base_concentration: float
        The base Na concentration prior to the simulation.
    er: float
        The relative permittivity of SiNx
    t_steps: int
        The number of time steps to simulate
    x_points_sin: int
        The number of grid points in the SiNx layer
    x_points_si: int
        The number of grid points in the Si layer
    out_dir: str
        The path to the output dir
    recovery_time: float
        Additional time used to model recovery (s). Default: 0.
    recovery_e_field: float
        Electric field applied during the recovery process in V/cm. Default: 0
    Returns
    -------
    str:
        The file_tag of the generated file
    """
    temperature_k = 273.15 + temperature_c
    d_sin = utils.evaluate_arrhenius(a0=d0_sinx, Ea=ea_sinx, temp=temperature_k)
    filetag = create_filetag(
        time_s=simulation_time,
        temp_c=temperature_c,
        sigma_s=sigma_s,
        zeta=zeta,
        d_sf=d_sf,
        ef=e_field,
        m=segregation_coefficient,
        h=h,
        recovery_time=recovery_time,
        recovery_e_field=recovery_e_field
    )
    params = {
        'file_tag': filetag,
        'time': simulation_time,
        'temperature': temperature_c,
        'sigma_s': '{0:.3E}'.format(sigma_s),
        'zeta': '{:.3E}'.format(zeta),
        'cb': '{0:.3E}'.format(base_concentration),
        'er': er,
        'tsteps': t_steps,
        'h': '{0:.3E}'.format(h),
        'm': '{0:.3E} '.format(segregation_coefficient),
        'd_sinx': '{:.2E}'.format(d_sin),
        'thickness_sinx': thickness_sin,
        'bias': '{0:.3E}'.format(sin_bias_from_e(e_field=e_field, thickness_sin=thickness_sin)),
        'xpoints_sinx': x_points_sin,
        'd_si': '{0:.3E}'.format(d_sf),
        'thickness_si': thickness_si,
        'xpoints_si': x_points_si,
        'recovery_time': recovery_time,
        'recovery_voltage': format(sin_bias_from_e(e_field=recovery_e_field, thickness_sin=thickness_sin)),
    }

    cwd = os.path.join(os.getcwd(), 'pnptransport')
    template_filename = os.path.join(cwd, template_file_fs)
    # open the template  file
    template_file = open(template_filename, 'r')
    # read it
    src = Template(template_file.read())
    result = src.substitute(params)
    template_file.close()

    fn = filetag + '.ini'
    output_filename = os.path.join(out_dir, fn)
    if platform == 'Windows':
        output_filename = r'\\?\\' + output_filename

    output_file = open(output_filename, 'w')
    output_file.write(result)
    output_file.close()
    current_date = datetime.now().strftime('%Y%m%d')
    append_to_batch_script(filetag=filetag, batch_script=os.path.join(out_dir, 'batch_' + current_date + '.sh'))
    return fn


def append_to_batch_script(filetag: str, batch_script: str):
    """
    Appends an execution line to the batch script

    Parameters
    ----------
    filetag: str
        The file tag for the .ini configuration file to run.
    batch_script: str
        The path to the batch script to append to.
    """
    cmd = './simulate_fs.py '
    cmd += '--config \'/home/fenics/shared/fenics/shared/simulations/input_finite_source/{0}\' &'.format(
        filetag + '.ini')
    if os.path.exists(batch_script):
        with open(batch_script, 'a', newline='\n') as file:
            file.write('\n' + cmd)
    else:
        with open(batch_script, 'w', newline='\n') as file:
            file.write('#!/bin/bash\n' + cmd)


def sigma_efield_variations(sigmas: np.ndarray, efields: np.ndarray, out_dir: str, zeta: float, simulation_time: float,
                            dsf: float, h: float, m: float, temperature_c: float, er: float = 7.0,
                            thickness_sin: float = 75E-3, thickness_si: float = 1., t_steps: int = 720,
                            x_points_sin: int = 100, x_points_si: int = 200, base_concentration: float = 1E-20):
    """
    Generates inputs for a combination of the initial surface concentrations and electric fields defined in the input.
    EVery other parameter remains fixed.

    Parameters
    ----------
    sigmas: np.ndarray
        An array containing the values of the surface concentration to vary (in ions/cm\ :sup:`2` \)
    efields: np.ndarray
        An array containing the values of the electric fields to vary (in V/cm)
    out_dir: str
        The path to the folder to store the generated input files.
    zeta: float
        The value of the rate of ingress at the surface (1/s)
    simulation_time: float
        The time length of the simulation in seconds.
    dsf: float
        The diffusion coefficient of Na in the stacking fault.
    h: float
        The surface mass transfer coefficient at the SiNx/Si interface in (cm/s)
    m: float
        The segregation coefficient at the SiNx/Si interface
    temperature_c: float
        The temperature °C
    er: float
        The relative permittivity of the dielectric. Default 7.0
    thickness_sin: float
        The thickness of the SiNx layer in um.
    thickness_si: float
        The thickness of the Si layer in um.
    t_steps: int
        The number of time steps. Default: 720
    x_points_sin: int
        The number of mesh points in the SiNx layer. Default 100
    x_points_si: int
        The number of mesh points in the Si layer. Default 200
    base_concentration: float
        The background concentration at the initial condition in atoms/cm\ :sup:`3` \

    """
    total_simulations = len(sigmas) * len(efields)
    # Create a data structure to index all the simulations
    ofat_simulations_db = np.empty(total_simulations, dtype=sim_db_dtype)

    if platform.system() == 'Windows':
        out_dir = r'\\?\\' + out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    counter = 0
    for s in sigmas:
        for e in efields:
            ini_file = create_input_file(
                simulation_time=simulation_time, temperature_c=temperature_c, sigma_s=s, zeta=zeta, d_sf=dsf,
                e_field=e, segregation_coefficient=m, h=h, thickness_sin=thickness_sin,
                thickness_si=thickness_si, base_concentration=base_concentration, er=er, t_steps=t_steps,
                x_points_sin=x_points_sin, x_points_si=x_points_si, out_dir=out_dir
            )
            bias = sin_bias_from_e(e_field=e, thickness_sin=thickness_sin)
            ofat_simulations_db[counter] = (
                ini_file, s, zeta, dsf, e, h, m, simulation_time, temperature_c, bias, 0.0, 0.0, 0.0, thickness_sin,
                thickness_si, er, base_concentration, t_steps, x_points_sin, x_points_si
            )

            counter += 1
            print('Created File for E: {0:.1E} V/cm, sigma: {1:.3E}'.format(e, s))

    # Save the simulations database to a csv file
    simulations_df = pd.DataFrame(ofat_simulations_db)
    simulations_df.to_csv(path_or_buf=os.path.join(out_dir, 'ofat_db.csv'), index=False)