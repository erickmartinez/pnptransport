import numpy as np
from scipy import interpolate
# from scipy import integrate
from pidsim.korol_conductivity import KorolConductivity
import pnptransport.transport_storage as data_storage
import pnptransport.utils as utils
import h5py
import os
# import tensorflow as tf
# from tensorflow import keras

class MLSim:
    """
    Attributes
    ----------
    _h5_transport_file: str
        The path to the h5 file containing the results from the transport simulation
    _data_storage: data_storage.TransportStorage
        A helper class to access the data in the h5 file
    __activated_na_fraction: float
        The fraction of Na atoms that contribute to as free careers to the conductivity model
    __segregation_coefficient: float
        The segregation coefficient of the stacking fault.
    __conductivity_cutoff: float
        The cutoff of the conductivity below which the shunt will be consider not existent in the x position.
        (defaults to 1E-10 S/cm)
    """
    _shunt_width: float = 0.57  # nm
    _shunt_length: float = 1.0 # 1.4142  # um
    _h5_transport_file: str = None
    _data_storage: data_storage.TransportStorage = None
    _conductivity_model: KorolConductivity = None
    __activated_na_fraction: float = 1.
    __segregation_coefficient: float = 1.
    __conductivity_cutoff = 1E-10  # S / cm
    __simulation_time: np.ndarray = None
    _cell_area: float = 1.0  # cm^2
    __ml_dump_mpp: str = None
    __ml_dump_rsh: str = None
    __dnn_dump_mpp: str = None
    __predictors_depth: np.ndarray = None
    __predictors_colnames: list = None

    def __init__(self, h5_transport_file: str):
        """
        Parameters
        ----------
        h5_transport_file: str
            The path to the h5 file containing the results of the transport simulation
        """
        self._h5_transport_file = h5_transport_file
        self._data_storage = data_storage.TransportStorage(filename=h5_transport_file)
        self._conductivity_model = KorolConductivity()
        self.__simulation_time = self._data_storage.time_s
        self.__ml_dump_mpp = os.path.join(os.getcwd(), 'pidsim/rfr_optimized_mpp.joblib')
        self.__ml_dump_rsh = os.path.join(os.getcwd(), 'pidsim/rfr_optimized_rsh.joblib')
        # self.__dnn_dump_mpp = os.path.join(os.getcwd(), 'pidsim/dnn_mpp.h5')
        self.__generate_predictors_depth()

    def __generate_predictors_depth(self):
        self.__predictors_depth = np.linspace(0.0, 1.0, 101)
        # self.__predictors_depth = utils.geometric_series_spaced(max_val=1.0, min_delta=1E-5, steps=99)
        self.__predictors_colnames = ['sigma at {0:.3f} um'.format(x) for x in self.__predictors_depth]
        self.__predictors_colnames.append('PNP depth')

    def set_mpp_dump_model(self, path: str):
        """
        Override the default ML regressor for the MPP data

        Parameters
        ----------
        path: str
            The path to the trained ML dump
        """
        self.__ml_dump_mpp = path

    def set_rsh_dump_model(self, path: str):
        """
        Override the default ML regressor for the Rsh data

        Parameters
        ----------
        path: str
            The path to the trained ML dump
        """
        self.__ml_dump_rsh = path

    @property
    def activated_na_fraction(self) -> float:
        return self.__activated_na_fraction

    @activated_na_fraction.setter
    def activated_na_fraction(self, value):
        value = abs(value)
        self._conductivity_model.activated_na_fraction = value
        self.__activated_na_fraction = value

    @property
    def segregation_coefficient(self):
        return self.__segregation_coefficient

    @segregation_coefficient.setter
    def segregation_coefficient(self, value: float):
        value = abs(value)
        self._conductivity_model.segregation_coefficient = value
        self.__segregation_coefficient = abs(value)

    @property
    def cell_area(self):
        return self._cell_area

    @cell_area.setter
    def cell_area(self, value: float):
        value = abs(value)
        if value == 0:
            raise ValueError('Trying to set the cell area to zero.')
        self._cell_area = value

    @property
    def conductivity_cutoff(self) -> float:
        return self.__conductivity_cutoff

    @conductivity_cutoff.setter
    def conductivity_cutoff(self, value: float):
        value = abs(value)
        self.__conductivity_cutoff = value

    @property
    def time_s(self) -> np.ndarray:
        return self.__simulation_time

    @property
    def h5_storage(self) -> data_storage.TransportStorage:
        return self._data_storage

    def get_requested_time_indices(self, requested_times: np.ndarray) -> np.ndarray:
        """
        Returns an array of the indices that match the requested times to the time array in file

        Parameters
        ----------
        requested_times: np.ndarray
            An array with the time points in seconds that we want to request the indices for.

        Returns
        -------
        np.ndarray:
            The array containing the indices that match the requested times to the time array in file
        """
        t_max = np.amax(self.time_s)
        if np.amax(requested_times) > t_max:
            raise ValueError('Trying to request points beyond the simulation time {0:.1f} h'.format(t_max))
        # Find the indices in the time series corresponding to the requested time points
        return utils.get_indices_at_values(x=self.time_s, requested_values=requested_times)

    def pmpp_time_series(self, requested_indices: np.ndarray) -> np.ndarray:
        """
        Gets the modeled Pmpp at the requested time points.

        Parameters
        ----------
        requested_indices: np.ndarray
            The time indices to estimate the Pmpp at.

        Returns
        -------
        np.ndarray:
            The time series for Pmpp
        """

        h5_path = self._data_storage.filename
        time_points = len(requested_indices)
        from tqdm import trange
        from sklearn.ensemble import RandomForestRegressor
        from joblib import load

        model_rf: RandomForestRegressor = load(self.__ml_dump_mpp)
        # model_dnn = tf.keras.models.load_model(self.__dnn_dump_mpp, save_format='h5')

        # allocate memory
        pmpp_t = np.empty(time_points)
        # Read the h5 file
        with h5py.File(h5_path, 'r') as hf:
            # Get the concentration dataset at the requested time
            pbar = trange(time_points, desc='Estimating Pmpp', leave=True, position=0)
            x1 = np.array(hf['/L1/x'])
            x2 = np.array(hf['/L2/x'])
            x2 = x2 - np.amax(x1)
            for i, v in enumerate(requested_indices):
                c = np.array(hf['/L2/concentration/ct_{0:d}'.format(v)])
                # Interpolate the concentration
                f = interpolate.interp1d(x=x2, y=c, kind='linear')
                # Get the concentration at the stacking fault partition
                concentration = f(self.__predictors_depth)
                self._conductivity_model.concentration_profile = concentration
                self._conductivity_model.segregation_coefficient = self.segregation_coefficient
                self._conductivity_model.activated_na_fraction = self.activated_na_fraction
                conductivity = self._conductivity_model.estimate_conductivity()
                # conductivity[conductivity < 1E-10] = 0.0
                # predictors = np.array([conductivity])
                predictors_list = list(conductivity)
                predictors_list.append(x2.max())
                predictors_list.append(self.__simulation_time[v])
                predictors = np.array([predictors_list])
                pmpp = np.array(model_rf.predict(X=predictors)).mean()
                # pmpp = np.array(model_dnn.predict(predictors)).mean()
                pmpp_t[i] = pmpp
                pbar.set_description('Time: {0:.1f} h, Pmpp = {1:.3f} mW/ cm^2'.format(
                    self.__simulation_time[v] / 3600, pmpp
                ))
                pbar.update(1)
                pbar.refresh()
        # rsh_t[rsh_t < 0] = np.amax(rsh_t)
        return pmpp_t

    def rsh_time_series(self, requested_indices: np.ndarray) -> np.ndarray:
        """
        Gets the modeled Pmpp at the requested time points.

        Parameters
        ----------
        requested_indices: np.ndarray
            The time indices to estimate the Pmpp at.

        Returns
        -------
        np.ndarray:
            The time series for Pmpp
        """

        h5_path = self._data_storage.filename
        time_points = len(requested_indices)
        from tqdm import trange
        from sklearn.ensemble import RandomForestRegressor
        from joblib import load

        model_rf = load(self.__ml_dump_rsh)

        # allocate memory
        rsh_t = np.empty(time_points)
        # Read the h5 file
        with h5py.File(h5_path, 'r') as hf:
            # Get the concentration dataset at the requested time
            pbar = trange(time_points, desc='Estimating Pmpp', leave=True, position=0)
            x1 = np.array(hf['/L1/x'])
            x2 = np.array(hf['/L2/x'])
            x2 = x2 - np.amax(x1)
            for i, v in enumerate(requested_indices):
                c = np.array(hf['/L2/concentration/ct_{0:d}'.format(v)])
                # Interpolate the concentration
                f = interpolate.interp1d(x=x2, y=c, kind='linear')
                # Get the concentration at the stacking fault partition
                concentration = f(self.__predictors_depth)
                self._conductivity_model.concentration_profile = concentration
                self._conductivity_model.segregation_coefficient = 1.
                self._conductivity_model.activated_na_fraction = 1
                conductivity = self._conductivity_model.estimate_conductivity()
                # predictors = np.array([conductivity])
                predictors_list = list(conductivity)
                predictors_list.append(x2.max())
                predictors_list.append(self.__simulation_time[v])
                predictors = np.array([predictors_list])
                rsh = 10 ** (np.array(model_rf.predict(X=predictors)).mean())
                rsh_t[i] = rsh
                pbar.set_description('Time: {0:.1f} h, Rsh = {1:.3f} Ohms cm^2'.format(
                    self.__simulation_time[v] / 3600, rsh
                ))
                pbar.update(1)
                pbar.refresh()
        # rsh_t[rsh_t < 0] = np.amax(rsh_t)
        return rsh_t

