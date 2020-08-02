import numpy as np
from scipy import interpolate
from scipy import integrate
from pidsim.korol_conductivity import KorolConductivity
import pnptransport.transport_storage as data_storage
import pnptransport.utils as utils
import h5py


class Rsh:
    """
    Attributes
    ----------
    _shunt_width: float
        The width of the stacking fault through which the shunt will be constructed.
        A default value of 0.57 nm is used in accordance to
        Volker Naumann, Dominik Lausch, Angelika Hähnel, Jan Bauer, Otwin Breitenstein, Andreas Graff, Martina Werner,
        Sina Swatek, Stephan Großer, Jörg Bagdahn, and Christian Hagendorf,  Sol. Energy Mater. Sol. Cells 120, 383
        (2014).
    _shunt_length: float
        The length of the shunt as projected on the SiNx/Si interface
    _h5_transport_file: str
        The path to the h5 file containing the results from the transport simulation
    _data_storage: data_storage.TransportStorage
        A helper class to access the data in the h5 file
    __activated_na_fraction: float
        The fraction of Na atoms that contribute to as free careers to the conductivity model
    __segregation_coefficient: float
        The segregation coefficient of the stacking fault.
    __n_resistors: int
        The number of resistors to partition the concentration profile in order to get Rsh
    __shunt_x: np.ndarray
        The x coordinate of the concentration profile in the silicon emitter
    __conductivity_cutoff: float
        The cutoff of the conductivity below which the shunt will be consider not existent in the x position.
        (defaults to 1E-10 S/cm)
    """
    _shunt_width: float = 0.57  # nm
    _shunt_length: float = 4  # um
    _h5_transport_file: str = None
    _data_storage: data_storage.TransportStorage = None
    _conductivity_model: KorolConductivity = None
    __activated_na_fraction: float = 1
    __segregation_coefficient: float = 50
    __n_resistors: int = 100
    __shunt_x: np.ndarray = None
    __shunt_partition: np.ndarray = None
    __shunt_depth: float = None
    __conductivity_cutoff = 1E-10  # S / cm
    __simulation_time: np.ndarray = None
    _cell_area = 1  # cm^2

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
        shunt_x = self._data_storage.get_position_data(layer=2)
        self.__shunt_x = shunt_x - np.amin(shunt_x)
        self.__simulation_time = self._data_storage.time_s

    @property
    def shunt_width(self) -> float:
        return self._shunt_width

    @shunt_width.setter
    def shunt_width(self, sw: float):
        self._shunt_width = abs(sw)

    @property
    def shunt_length(self) -> float:
        return self._shunt_length

    @shunt_length.setter
    def shunt_length(self, sl):
        self._shunt_length = abs(sl)

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
    def n_resistors(self) -> int:
        return self.__n_resistors

    @n_resistors.setter
    def n_resistors(self, value):
        value = abs(int(value))
        if value == 0:
            raise ValueError('The number of resistors must be an integer greater than 0.')
        self.__n_resistors = value

    @property
    def shunt_area(self):
        return self._shunt_width * self._shunt_length * 1E-11

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
    def profile_x(self) -> np.ndarray:
        return self.__shunt_x
    
    @property
    def shunt_depth(self) -> float:
        return np.amax(self.profile_x)

    @property
    def time_s(self) -> np.ndarray:
        return self.__simulation_time

    @property
    def _x_partition(self) -> np.ndarray:
        """
        Partitions the original x coordinates of the concentration profile into a new set of points to center the
        series resitors at.
        """
        if self.__shunt_partition is None:
            self.__shunt_partition = np.linspace(0.0, 1.0)
        return self.__shunt_partition

    def resistance_at_time_t(self, time_s: float):
        """
        Estimates the resistance of the concetration profile as the equivalent resistance of a segment of series
        resistors corresponding to a partition of the concentration profile into parallelepipeds
        Parameters
        ----------
        time_s

        Returns
        -------

        """
        partition_x = self._x_partition
        # Get the concentration profile at time ts
        c = self._data_storage.get_concentration_at_time_t(requested_time_s=time_s, layer=2)
        # Interpolate the concentration
        f = interpolate.interp1d(x=self.__shunt_x, y=c, kind='linear')
        # Get the concentration at the stacking fault partition
        concentration = f(partition_x)
        self._conductivity_model.concentration_profile = concentration
        self._conductivity_model.segregation_coefficient = 50
        conductivity = self._conductivity_model.estimate_conductivity()
        # conductivity = conductivity[conductivity >= self.__conductivity_cutoff]
        resistivty = 1 / conductivity
        resistance = np.sum(resistivty) * self._shunt_length * 1E-4 / self.shunt_area

        return resistance

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

    def resistance_time_series(self, requested_indices: np.ndarray) -> np.ndarray:
        """
        Gets the modeled Rsh at the requested time points.

        Parameters
        ----------
        requested_indices: np.ndarray
            The time indices to estimate the Rsh at.

        Returns
        -------
        np.ndarray:
            The time series for Rsh
        """

        partition_x = self._x_partition
        h5_path = self._data_storage.filename
        time_points = len(requested_indices)
        from tqdm import trange

        # allocate memory
        rsh_t = np.empty(time_points)
        # Read the h5 file
        with h5py.File(h5_path, 'r') as hf:
            # Get the concentration dataset at the requested time
            pbar = trange(time_points, desc='Estimating Rsh', leave=True)
            for i, v in enumerate(requested_indices):
                c = np.array(hf['/L2/concentration/ct_{0:d}'.format(v)])
                # Interpolate the concentration
                f = interpolate.interp1d(x=self.__shunt_x, y=c, kind='linear')
                # Get the concentration at the stacking fault partition
                concentration = f(partition_x)
                self._conductivity_model.concentration_profile = concentration
                self._conductivity_model.segregation_coefficient = 50
                self._conductivity_model.activated_na_fraction = 1
                conductivity = self._conductivity_model.estimate_conductivity()
                idx_length = conductivity >= self.__conductivity_cutoff

                # if len(partition_x[idx_length]) > 0:
                #     shunt_depth = np.amax(partition_x[idx_length])
                #     conductivity = conductivity[idx_length]
                #     partition_x = partition_x[idx_length]
                # else:
                #     shunt_depth = 0
                #     conductivity = np.array([1])
                #     partition_x = np.array([1])
                resistivty = 1 / conductivity
                dx = partition_x[1] - partition_x[0]
                rsh = np.sum(resistivty) * dx  #/ (self._shunt_width * 1E-7) / self._shunt_length
                # rsh = integrate.simps(y=(resistivty[0::]/(self._shunt_length - partition_x[0::])), x=partition_x[0::])
                # rsh *= self.cell_area
                rsh = rsh# / (self._shunt_length * self._shunt_width * 1E-7)
                # rsh *= np.sqrt(3)/(2*self._shunt_width * 1E-7)
                rsh_t[i] = rsh
                pbar.set_description('Time: {0:.1f} h, Rsh = {1:.3g} Ohm cm^2'.format(
                    self.__simulation_time[v]/3600, rsh
                ))
                pbar.refresh()
        # rsh_t[rsh_t < 0] = np.amax(rsh_t)
        print(rsh_t)
        return rsh_t
