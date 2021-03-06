import numpy as np
from scipy import interpolate
# from scipy import integrate
from pidsim.korol_conductivity import KorolConductivity
import pnptransport.transport_storage as data_storage
import pnptransport.utils as utils
import h5py


class Rsh:
    """
    This class provides methods to estimate the shunt resistance :math:`R_{\mathrm{sh}}` of a Na concentration profile
    in a Si SF defect. It assumes the defect is a rectangular structure of 0.57 nm of width and that the concentration
    can be discretized within parallelepipeds within the stacking fault, in which the Na concentration is mapped to
    a resistivity. Then the resistivities are added in series

    .. math:: R_{\mathrm{Na}} = \sum_{i} R_i

    We assume that :math:`R_{\mathrm{Na}}` is in parallel with the junction resistance :math:`R_{\mathrm{jun}}`.

    The shunt resistance is estimated then by

    .. math:: R_{\\mathrm{sh}} = \\frac{R_{\mathrm{Na}} R_{\\mathrm{jun}}}{R_{\\mathrm{Na}} + R_{\\mathrm{jun}}}

    Currently it uses maps the conductivity values to resistivity using KorolConductivity model.

    *Example*

    .. code-block:: python

        import pidsim.rsh as prsh
        import numpy as np
        path_to_h5 = './transport_simulation_output.h5'
        requested_times = np.linspace(0, 3600, 50)
        rsh_analysis = prsh.Rsh(h5_transport_file=path_to_h5)
        # Get the indices of the respective time points in the h5 file:
        requested_indices = prsh.get_requested_time_indices(requested_times=requested_times)
        rsh = prsh.resistance_time_series(requested_indices=requested_indices)


    Attributes
    ----------
    _shunt_width: float
        The width of the stacking fault through which the shunt will be constructed.
        A default value of 0.57 nm is used in accordance to

        Volker Naumann, Dominik Lausch, Angelika Hähnel, Jan Bauer, Otwin Breitenstein, Andreas Graff, Martina Werner,
        Sina Swatek, Stephan Großer, Jörg Bagdahn, and Christian Hagendorf,  *Sol. Energy Mater. Sol. Cells* **120**,
        83 (2014).
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
    _shunt_length: float = 1.4142  # um
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
    __rsh_0: float = 1E5
    __max_depth: float = 1

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
    def h5_storage(self) -> data_storage.TransportStorage:
        return self._data_storage

    @property
    def rsh_0(self) -> float:
        return self.__rsh_0

    @rsh_0.setter
    def rsh_0(self, value: float):
        value = abs(value)
        if value == 0:
            raise ValueError('Trying to set rsh(t=0) to 0.')
        self.__rsh_0 = value

    @property
    def _x_partition(self) -> np.ndarray:
        """
        Partitions the original x coordinates of the concentration profile into a new set of points to center the
        series resitors at.
        """
        if self.__shunt_partition is None:
            self.__shunt_partition = np.linspace(0.0, self.__max_depth)
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
        float:
            The resistance in Ohm cm
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
            pbar = trange(time_points, desc='Estimating Rsh', leave=True, position=0)
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
                # idx_length = conductivity >= self.__conductivity_cutoff

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
                rsh = np.sum(resistivty) * dx * 1E-4  # / (self._shunt_width * 1E-7) / self._shunt_length
                # rsh = integrate.simps(y=(resistivty[0::]/(self._shunt_length - partition_x[0::])), x=partition_x[0::])
                # rsh *= self.cell_area
                # rsh = rsh / (self._shunt_length * self._shunt_width * 1E-7)
                # rsh *= np.sqrt(3)/(2*self._shunt_width * 1E-7)
                rsh_t[i] = rsh * self.__rsh_0 * self.cell_area / (self.__rsh_0 * self.cell_area + rsh)
                pbar.set_description('Time: {0:.1f} h, Rsh = {1:.3g} Ohm cm^2'.format(
                    self.__simulation_time[v] / 3600, rsh
                ))
                pbar.update(1)
                pbar.refresh()
        # rsh_t[rsh_t < 0] = np.amax(rsh_t)
        return rsh_t

    def interface_concentrations_time_series(self, requested_indices: np.ndarray) -> np.ndarray:
        """
        Returns the values of the concentration at both sides of the SiNx/Si interface

        Parameters
        ----------
        requested_indices: np.ndarray
            The time indices to estimate the concentrations at.

        Returns
        -------
        np.ndarray
            A data structure containing t, CSiNx, CSi
        """
        h5_path = self._data_storage.filename
        time_points = len(requested_indices)
        from tqdm import trange

        # allocate memory
        c_data = np.empty(time_points, np.dtype([('time_s', 'd'), ('C_SiNx (cm^-3)', 'd'), ('C_Si (cm^-3)', 'd')]))
        # Read the h5 file
        with h5py.File(h5_path, 'r') as hf:
            pbar = trange(time_points, desc='Estimating Rsh', leave=True)
            for i, v in enumerate(requested_indices):
                c1 = np.array(hf['/L1/concentration/ct_{0:d}'.format(v)])
                c2 = np.array(hf['/L2/concentration/ct_{0:d}'.format(v)])
                c_data[i] = (self.__simulation_time[v], c1[-1], c2[0])
                pbar.set_description('{0:.1f} h, C1 = {1:.3E}, C2 = {2:.3E} cm^-3.'.format(
                    self.__simulation_time[v] / 3600, c1[-1], c2[0]
                ))
                pbar.update(1)
                pbar.refresh()
        return c_data

    def dielectric_flux_time_series(self, requested_indices: np.ndarray) -> np.ndarray:
        """
        Returns the values of the fluxes at the source and at the middle of the dielectric

        Parameters
        ----------
        requested_indices: np.ndarray
            The time indices to estimate the fluxes at.

        Returns
        -------
        np.ndarray
            A data structure containing t, Js, Jd
        """
        h5_path = self._data_storage.filename
        time_points = len(requested_indices)
        from tqdm import trange

        # allocate memory
        j_data = np.empty(time_points, np.dtype([('time_s', 'd'), ('J_1 (cm/s)', 'd'), ('J_2 (cm/s)', 'd')]))
        # Read the h5 file
        with h5py.File(h5_path, 'r') as hf:
            pbar = trange(time_points, desc='Estimating Rsh', leave=True)
            grp_sinx = hf['/L1']
            x1 = np.array(grp_sinx['x'])
            # find the middle point of the SiNx layer
            idx_middle = int((np.abs(x1 - 0.5 * np.amax(x1))).argmin())
            dx1 = (x1[1] - x1[0]) * 1E-4  # um to cm
            dx2 = (x1[idx_middle + 1] - x1[idx_middle]) * 1E-4  # um to cm
            # Try to see if h0 is defined (source limited)
            time_metadata = self._data_storage.get_metadata(group='/time')
            source_limited = False
            if 'h0' in time_metadata:
                source_limited = True
                xs = time_metadata['c_surface'] / time_metadata['cs_0']
            for i, v in enumerate(requested_indices):
                c1 = np.array(hf['/L1/concentration/ct_{0:d}'.format(v)])
                p1 = np.array(hf['/L1/potential/vt_{0:d}'.format(v)])
                D1 = float(grp_sinx.attrs['D'])
                mu1 = float(grp_sinx.attrs['ion_mobility'])

                e_field_1 = (p1[0] - p1[1])/dx1
                e_field_2 = (p1[idx_middle] - p1[idx_middle + 1]) / dx2
                # Flux around the source
                # If source limited determine the flux from h0
                if source_limited:
                    j1 = time_metadata['cs_0']*np.exp(-time_metadata['h0'] * self.__simulation_time[v] / xs)
                else:
                    j1 = D1 * (c1[0] - c1[1]) / dx1 + 0.5 * (c1[0] + c1[1]) * mu1 * e_field_1
                j2 = D1 * (c1[idx_middle] - c1[idx_middle + 1]) / dx2
                j2 += 0.5 * (c1[idx_middle] + c1[idx_middle + 1]) * mu1 * e_field_2

                j_data[i] = (self.__simulation_time[v], j1, j2)
                pbar.set_description('{0:.1f} h, j_1 = {1:.3E}, j_2 = {2:.3E}'.format(
                    self.__simulation_time[v] / 3600, j1, j2
                ))
                pbar.update(1)
                pbar.refresh()
        return j_data
