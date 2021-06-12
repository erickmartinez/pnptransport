import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import os
import platform
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy import integrate
import pnptransport.utils as utils
from tqdm import tqdm
from tqdm import trange
import json
from joblib import load
from sklearn.neural_network import MLPRegressor

sentaurus_db = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\sentaurus_ml_db.csv'
hidden_layers = 10
output_folder = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results'
nn_dump_mpp = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\mlp_optimized_mpp_5_hl.joblib'
nn_dump_rsh = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\mlp_optimized_rsh_5_hl.joblib'
# nn_dump_mpp = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\gbr_optimized_mpp.joblib'
# nn_dump_rsh = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\gbr_optimized_rsh.joblib'
# nn_dump_mpp = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\rfr_optimized_mpp_20201118.joblib'
# nn_dump_rsh = r'G:\My Drive\Research\PVRD1\FENICS\SUPG_TRBDF2\simulations\sentaurus_fitting\results\rfr_optimized_rsh_20201118.joblib'


if __name__ == '__main__':
    if platform.system() == 'Windows':
        output_folder = r'\\?\\' + output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Load my style
    with open('plotstyle.json', 'r') as style_file:
        mpl.rcParams.update(json.load(style_file)['defaultPlotStyle'])

    df = pd.read_csv(sentaurus_db)
    n_c_points = 101
    # The maximum depth in um to take for the concentration profile
    x_max = 1.
    x_inter = np.linspace(start=0., stop=x_max, num=n_c_points)
    features_cols = ['sigma at {0:.3E} um'.format(x) for x in x_inter]
    features_cols.append('PNP depth')
    features_cols.append('time (s)')

    column_list_pmpp = list(set(list(df.columns)) - set(['Rsh (Ohms cm2)']))#, 'time (s)']))
    # If fitting rsh uncomment the next line
    column_list_rsh = list(set(list(df.columns)) - set(['pd_mpp (mW/cm2)']))#, 'time (s)']))
    column_list_pmpp.sort()
    column_list_rsh.sort()
    df_mpp = df[column_list_pmpp]
    df_rsh = df[column_list_rsh]

    target_column_mpp = ['pd_mpp (mW/cm2)']
    # If fitting rsh uncomment the next line
    target_column_rsh = ['Rsh (Ohms cm2)']
    # predictors = list(set(list(df_mpp.columns)) - set(target_column_mpp))
    predictors = features_cols

    rsh_sentaurus = np.array(df['Rsh (Ohms cm2)'])
    mpp_sentaurus = np.array(df['pd_mpp (mW/cm2)'])
    model_mpp: MLPRegressor = load(nn_dump_mpp)
    # model_rsh: MLPRegressor = load(nn_dump_rsh)

    for f in predictors:
        print(f)

    X = df[predictors].values
    y_mpp = df['pd_mpp (mW/cm2)'].values.ravel()
    # y_rsh = np.log(df['Rsh (Ohms cm2)'].values.ravel())

    [n_examples, n_features] = X.shape

    y_mpp_pred = np.empty(n_examples)
    y_rsh_pred = np.empty(n_examples)

    # pbar = trange(n_examples, desc='Estimating Pmpp and Rsh', leave=True, position=0)

    # for i in range(n_examples):
    #     x = np.array([X[i]])
    #     y_mpp_pred[i] = model_mpp.predict(X=x)
    #     y_rsh_pred[i] = model_rsh.predict(X=x)
    #     pbar_desc = 'Example MPP: {0:.3f}, Prediction MPP = {1:.3f} mW/ cm^2 '.format(y_mpp[i], y_mpp_pred[i])
    #     # pbar_desc += 'Example RSH: {0:.3f}, Prediction RSH = {1:.3f} (Log[Ohm cm^2])'.format(y_rsh[i], y_rsh_pred[i])
    #     pbar.set_description(pbar_desc)
    #     pbar.update(1)
    #     pbar.refresh()
    y_mpp_pred = model_mpp.predict(X=X)
    # y_rsh_pred = model_rsh.predict(X=X)
    y_mpp_squared_error = np.power(y_mpp_pred - y_mpp, 2.)
    df['y_mpp_squared_error'] = y_mpp_squared_error
    folder_series = pd.Series(df['Folder'])
    df['finite source'] = folder_series.str.startswith('FS').astype(bool)
    large_error_simulations = df[df['y_mpp_squared_error'] > 1].reset_index()
    df.to_csv(os.path.join(output_folder, 'sentaurus_ml_db.csv'))
    print(large_error_simulations.columns[-4::])
    large_error_simulations = large_error_simulations[[
        'Folder', 'PNP depth', 'time (s)', 'finite source', 'y_mpp_squared_error',
        'pd_mpp (mW/cm2)'
    ]]
    finite_source_df = df[df['finite source']]

    idx_large_error = y_mpp_squared_error > 1
    idx_finite_source = np.array(df['finite source'], dtype=bool)
    large_error_simulations['predicted pd_mpp (mW/cm^2)'] = y_mpp_pred[idx_large_error]
    large_error_simulations.to_csv(os.path.join(output_folder, 'large_error_nn.csv'), index=False)

    fig = plt.figure(1)
    fig.set_size_inches(3.5, 3.5, forward=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gs_0 = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
    gs_00 = gridspec.GridSpecFromSubplotSpec(
        nrows=1, ncols=1, subplot_spec=gs_0[0], hspace=0.1,
    )
    ax_0 = fig.add_subplot(gs_00[0, 0])
    # ax_1 = fig.add_subplot(gs_00[1, 0])
    ax_0.set_aspect('equal', 'box')
    # ax_1.set_aspect('equal', 'box')
    # Set the axis labels
    ax_0.set_xlabel(r'MPP Sentaurus ($\mathrm{mW/cm^2}$)')
    ax_0.set_ylabel(r'MPP MLP ($\mathrm{mW/cm^2}$)')

    # ax_1.set_xlabel(r'RSH Sentaurus ($\mathrm{\Omega\cdot  cm^2}$)')
    # ax_1.set_ylabel(r'RSH MLP ($\mathrm{\Omega\cdot  cm^2}$)')

    # ax_1.set_xscale('log')
    # ax_1.set_yscale('log')

    ax_0.plot(
        y_mpp, y_mpp_pred, ls='None', color='C0', marker='o', zorder=0
    )

    ax_0.plot(
        y_mpp[idx_large_error], y_mpp_pred[idx_large_error], ls='None', color='r', marker='x',
        label='MSE > 1', zorder=1
    )

    ax_0.plot(
        y_mpp[idx_finite_source], y_mpp_pred[idx_finite_source], ls='None', color='tab:purple', marker='*',
        label='Finite Source', zorder=2
    )

    leg = ax_0.legend(loc='best', frameon=True)

    # ax_1.plot(
    #     y_rsh, y_rsh_pred, ls='None', color='C1', marker='o'
    # )

    ax_0.set_xlim(left=-1, right=y_mpp.max())
    # ax_1.set_xlim(left=0, right=y_rsh.max())

    ax_0.set_ylim(bottom=-1, top=y_mpp.max())
    # ax_1.set_ylim(bottom=0, top=y_rsh.max())

    fig.tight_layout()
    plt.show()

    fig.savefig(
        os.path.join(output_folder, 'mlp_5_hl_vs_sentaurus.png'), dpi=300
    )
