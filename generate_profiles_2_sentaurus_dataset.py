"""
This code iterates over the Sentaurus device simulations folder to look for
1. Efficiency time series data
2. Rsh time series data
3. Conductivity profile time series

It also looks for the h5 file corresponding to the transport simulation in pidlogger, tries to find the file within
a given file path, checking that the length of the profile points matches the length of the depth dataset in the
transport simulation dataset.

It interpolates the corresponding profile within a distance of 1.0 um and appends a dataset of the following shape

+---+--------+-----------+-----+----------+----------+-----------------+----------------+
| n | s(x=0) | s(x=0.01) | ... | s(x=1.0) | time (s) | pd_mpp (mW/cm2) | Rsh (Ohms cm2) |
+---+--------+-----------+-----+----------+----------+-----------------+----------------+

@author Erick R Martinez Loran <erm013@ucsd.edu>
2020
"""
import numpy as np
import pandas as pd
import os
import platform
from scipy import interpolate
import h5py
import glob
import re
from shutil import copyfile


base_path = r'G:\Shared drives\FenningLab2\Projects\PVRD1\Simulations\Sentaurus PID\results\3D'
output_folder = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting'
pnp_base = r'G:\My Drive\Research\PVRD1\Sentaurus_DDD\pnp_simulations'


def find_h5(the_path: str, the_file: str, len_sigma: int):
    files = glob.glob('{0}\**\{1}'.format(the_path, the_file))
    if len(files) > 0:
        for k, f_ in enumerate(files):
            with h5py.File(f_, 'r') as hf:
                len_x2 = len(np.array(hf['L2/x']))
            if len_sigma == len_x2:
                print('The length of x2 in file {0} matches the length of the conductivity dataset ({1}).'.format(f_, len_sigma))
                return files[k]

    return None


if __name__ == '__main__':
    # If the system is Windows prepend the paths with \\?\\ to correctly find long paths
    if platform.system() == 'Windows':
        base_path = r'\\?\\' + base_path
        output_folder = r'\\?\\' + output_folder

    # If the output path does not exist, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of subfolders in the base path (each subfolder has a single result of a sentaurus simulation)
    folder_list = os.listdir(base_path)

    # Iterate over the folder list to estimate the total number of rsh points to store
    n_rsh = 0
    # The number of interpolated conductivity points to select
    n_c_points = 100
    # The maximum depth in um to take for the concentration profile
    x_max = 1.
    x_inter = np.linspace(start=0., stop=1., num=n_c_points)
    column_names = ['sigma at {0:.3f} um'.format(x) for x in x_inter]
    column_names.append('time (s)')
    column_names.append('pd_mpp (mW/cm2)')
    column_names.append('Rsh (Ohms cm2)')

    # A regular expression to find the names of the h5 files corresponding to the transport simulation
    pattern = re.compile(r".*\/(.*\.h5)")

    df = pd.DataFrame(columns=column_names)
    for fb in folder_list:
        f = os.path.join(base_path, fb)

        efficiency_file = os.path.join(f, r'jv_plots\efficiency_results.csv')
        rsh_file = os.path.join(f, r'analysis_plots\rsh_data.csv')
        conductivity_file = os.path.join(f, 'conductivity_profiles.h5')
        # Check that the files exist
        efficiency_file_exists = os.path.exists(efficiency_file)
        rsh_file_exists = os.path.exists(rsh_file)
        conductivity_file_exists = os.path.exists(conductivity_file)

        # Get the name of the h5 file from the pidlog
        pidlog_file = os.path.join(f, 'pidlogger.log')
        h5_file = ''
        try:
            with open(pidlog_file, 'r') as pf:
                for line in pf:
                    m = re.match(pattern, line)
                    if m is not None:
                        h5_file = m[1]
                        break
        except Exception as e:
            print('Could not find {0}'.format(pidlog_file))
            continue

        # Get the length of the conductivity dataset
        ds_name = 'sigma_0'
        with h5py.File(conductivity_file, 'r') as hf:
            # Get the conductivity data set
            n_sigma = len(np.array(hf['/conductivity'][ds_name]))

        # find the h5 file
        path_to_h5 = find_h5(pnp_base, h5_file, len_sigma=n_sigma)
        # move the file to the sentaurus simulations folder
        # copyfile(src=path_to_h5, dst=os.path.join(f, 'pnp_transport.h5'))

        if efficiency_file_exists and rsh_file_exists and conductivity_file_exists and (path_to_h5 is not None):
            print('Analyzing folder {0}'.format(f))
            # Get the list of time points in the h5 file
            with h5py.File(path_to_h5, 'r') as hf:
                time_s = np.array(hf['time'])
                x1 = np.array(hf['L1/x'])
                x2 = np.array(hf['L2/x'])
                depth = x2 - np.amax(x1)
            # Read the efficiency and Rsh files
            efficiency_df = pd.read_csv(efficiency_file)
            rsh_file_df = pd.read_csv(rsh_file)
            rsh_file_df['time (s)'] = rsh_file_df['time (h)'] * 3600
            merged_df = pd.merge(efficiency_df, rsh_file_df, on='time (s)', how='inner')
            required_columns = ['time (s)', 'pd_mpp (mW/cm2)', 'Rsh (Ohms cm2)']
            merged_df = merged_df[required_columns]
            # Iterate over the merged df to get the time and find the respective concentration profile
            for i, r in merged_df.iterrows():
                time_i = r['time (s)']
                # find the index of the corresponding time point to later locate the conductivity profile at that time
                idx = np.abs(time_i - time_s).argmin()
                # construct the dataset name
                ds_name = 'sigma_{0}'.format(idx)
                with h5py.File(conductivity_file, 'r') as hf:
                    # Get the conductivity data set
                    sigma = np.array(hf['/conductivity'][ds_name])

                # Interpolate the dataset
                # The number of columns in the dataset
                n_cols = len(x_inter) + 3
                if len(sigma) == len(depth):
                    f = interpolate.interp1d(depth, sigma)
                    sigma_interp = f(x_inter)
                    data_i = np.zeros(n_cols)
                    for j in range(len(sigma_interp)):
                        data_i[j] = sigma_interp[j]
                    data_i[j+1] = time_i
                    data_i[j+2] = r['pd_mpp (mW/cm2)']
                    data_i[j+3] = r['Rsh (Ohms cm2)']

                    data_to_append = {}
                    for j, col in enumerate(column_names):
                        data_to_append[col] = data_i[j]
                    df = df.append(data_to_append, ignore_index=True)
                    # print(df)

                # except Exception as e:
                #     print(e)
                #     continue
    print(df)
    df.to_csv(os.path.join(output_folder, 'sentaurus_ml_db.csv'), index=False)


        
