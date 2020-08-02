import numpy as np
import pnptransport.hd5storage as h5storage


class TransportStorage(h5storage.H5Storage):
    __layers = [1, 2]
    __time_s: np.ndarray = None

    def __init__(self, filename: str):
        super().__init__(filename=filename)

    def create_time_dataset(self, time_s: np.ndarray):
        self.create_dataset(name='time', data=time_s)

    @property
    def time_s(self) -> np.ndarray:
        """
        Queries the time dataset

        Returns
        -------
        np.ndarray:
            The time dataset in seconds
        """
        if self.__time_s is None:
            self.__time_s = self.get_numpy_dataset(name='time')
        return self.__time_s

    def get_position_data(self, layer: int) -> np.ndarray:
        """
        Gets the x coordinate for the transport simulation in the layer X

        Parameters
        ----------
        layer: int
            The layer to query the position data from (1 or 2)

        Returns
        -------
        np.ndarray:
            An array containing the coordinates for the layer n
        """
        if layer not in self.__layers:
            raise IndexError('Layer {0} out of bounds. Valid layers are 1 or 2'.format(layer))

        return self.get_numpy_dataset(name='x', group_name='/L{0}'.format(int(layer)))

    def get_concentration_at_time_t(self, requested_time_s: float, layer: int) -> np.ndarray:
        """
        Tries to find the concentration profile at the specified time

        Parameters
        ----------
        requested_time_s: Union[float, int]
            The time at which we want to retrive the concentration profile
        layer: int
            The layer from which to get the concentration profile (valid values are 1 and 2)
        Returns
        -------
        np.ndarray:
            The concentration profile in atoms/cm^3
        """
        if layer not in self.__layers:
            raise IndexError('Layer {0} out of bounds. Valid layers are 1 or 2'.format(layer))

        time_s = self.time_s
        # Find the index closet to the requested time
        idx = (np.abs(requested_time_s - time_s)).argmin()
        dataset_name = 'ct_{0:d}'.format(int(idx))
        return self.get_numpy_dataset(name=dataset_name, group_name='/L{0:d}/concentration'.format(int(layer)))

    def get_potential_at_time_t(self, requested_time_s: float) -> np.ndarray:
        """
        Tries to find the concentration profile at the specified time

        Parameters
        ----------
        requested_time_s: Union[float, int]
            The time at which we want to retrive the concentration profile

        Returns
        -------
        np.ndarray:
            The concentration profile in atoms/cm^3
        """

        time_s = self.time_s
        # Find the index closet to the requested time
        idx = (np.abs(requested_time_s - time_s)).argmin()
        dataset_name = 'ct_{0:d}'.format(int(idx))
        return self.get_numpy_dataset(name=dataset_name, group_name='/L1/concentration')
