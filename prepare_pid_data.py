import numpy as np
import pandas as pd
import os

base_path = r'G:\My Drive\Research\PVRD1\DATA\PID'
csv_file = r'PID_mc_BSF_4.csv'


def hhmmss_2_seconds(hhmmss):
    thms = hhmmss.split(':')
    ts = 0
    for i, v in enumerate(thms):
        j = 2 - i
        f = 60 ** j
        ts += f * int(v)
    return ts


if __name__ == '__main__':
    full_file = os.path.join(base_path, csv_file)
    df = pd.read_csv(full_file, converters={'Time [hh:mm]': hhmmss_2_seconds})
    df = df.rename(columns={'Time [hh:mm]': 'time (s)'})
    new_name = os.path.splitext(os.path.basename(csv_file))[0]
    new_name = '{0}_ready.csv'.format(new_name)
    df.to_csv(path_or_buf=os.path.join(base_path, new_name), index=False)
