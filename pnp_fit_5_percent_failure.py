import numpy as np
import pandas as pd
from scipy import interpolate
import os

base_path = r'G:\My Drive\Research\PVRD1\Manuscripts\Device_Simulations_draft\simulations\inputs_20200831'
pi4_dtype = np.dtype([
    ('pi_1', 'd'),
    ('pi_2', 'd'),
    ('pi_3', 'd'),
    ('pi_4', 'd')
])
# The D0 for sinx
_d0_sinx = 1E-14
# The activation energy for sinx
_ea_sinx = 0.1


if __name__ == '__main__':
    ofat_df = pd.read_csv(os.path.join(base_path, 'ofat_db.csv'), index_col=None)
    ofat_rows = len(ofat_df)
    pi_data = np.empty(ofat_rows, dtype=pi4_dtype)
    tau_5 = np.empty(ofat_rows)
    for i, row in ofat_df.iterrows():
        if bool(row['converged']):
            config_file = row['config file']
            file_tag = os.path.splitext(os.path.basename(config_file))[0]
            pid_file = file_tag + '_simulated_pid.csv'
            csv_file = os.path.join(base_path, 'batch_analysis', pid_file)
            pid_df = pd.read_csv(csv_file)
            time_s = pid_df['time (s)'].to_numpy(dtype=float)
            power = pid_df['Pmpp (mW/cm^2)'].to_numpy(dtype=float)
            rsh = pid_df['Rsh (Ohm cm^2)'].to_numpy(dtype=float)
            time_h = time_s / 3600.

            t_interp = np.linspace(np.amin(time_s), np.amax(time_s), num=1000)
            f_p_interp = interpolate.interp1d(time_s, power, kind='linear')
            power_interp = f_p_interp(t_interp)
            idx_5 = (np.abs(power_interp / power_interp[0] - 0.95)).argmin()
            tau_5[i] = t_interp[idx_5].copy()
            S0 = row['sigma_s (cm^-2)']
            k = row['zeta (1/s)']
            h = row['h (cm/s)']
            temperature_c = row['']
            temperature_k = 273.15 + temperature_c
            d_sin = utils.evaluate_arrhenius(a0=_d0_sinx, Ea=_ea_sinx, temp=temperature_k)
            pi_1 = float(row['zeta (1/s)']) * tau_5[i]
            pi_2 = float(row[''])