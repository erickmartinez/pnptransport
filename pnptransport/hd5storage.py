import numpy as np
import h5py


class H5Storage:
    """
    This class provides abstraction for creating, updating and retriving information from h5 files

    Attributes
    ----------
    _filename: str
        The path to the h5 datafile

    Methods
    -------
    create_storage:
        Creates the
    """
    _filename: str = None

    def __init__(self, filename: str):
        """
        Parameters
        ----------
        filename: str
            The name of the h5 file to store data to.
        """
        self._filename = filename

    def create_storage(self):
        """
        Creates the h5 file and appends the basic groups
        """
        with h5py.File(self._filename, 'w') as hf:
            hf.create_group(name='L1')
            hf.create_group(name='L2')

    def metadata(self, metadata: dict, group="/"):
        """
        Saves a dictionary with the measurement metadata to the specified dataset/group.

        Parameters
        ----------
        metadata: dict
            A dictionary with the metadata to save
        group: str
            The dataset/group to save the attributes to.
        """
        if not isinstance(metadata, dict):
            raise TypeError('The argument must be of type ')
        with h5py.File(self._filename, 'a') as hf:
            group = hf.get(group) if group != "/" else hf
            for key, val in metadata.items():
                group.attrs[key] = val

    def get_metadata(self, group="/") -> dict:
        """
        Returns the attributes of a selected group.

        Parameters
        ----------
        group: str
            The group to get the attributes from

        Returns
        -------
        dict:
            A dictionary with the attributes of the dataset/group
        """
        with h5py.File(self._filename, 'r') as hf:
            metadata = dict(hf.get(group).attrs)
        return metadata

    def create_dataset(self, name: str, data: np.ndarray, group_name: str = "/"):
        """
        Creates a non-resizable dataset in the group 'group_name'.

        Parameters
        ----------
        name: str
            The name of the dataset
        data: np.ndarray
            The data to store
        group_name: str
            The name of the group to save the dataset to
        """
        if not isinstance(name, str):
            raise TypeError('Name should be an instance of str')
        with h5py.File(self._filename, 'a') as hf:
            if group_name not in hf:
                hf.create_group(group_name)
            ds_shape = data.shape
            group = hf.get(group_name)
            group.create_dataset(name=name, shape=ds_shape, dtype=data.dtype, compression='gzip', data=data)

    def create_group(self, name: str, parent_group: str = None):
        """

        Parameters
        ----------
        name: str
            The name of the group to create
        parent_group: str
            The parent group

        """
        with h5py.File(self._filename, 'a') as hf:
            if parent_group is None:
                hf.create_group(name)
            else:
                pg = hf[parent_group]
                pg.create_group(name)

    def get_numpy_dataset(self, name: str, group_name: str = "/") -> np.ndarray:
        """
        Retrives the dataset from the selected group in the h5 file

        Parameters
        ----------
        name: str
            The name of the dataset to extract
        group_name: str
            The name of the group the dataset is at (default '/')

        Returns
        -------
        np.ndarray
            The requested datast
        """
        with h5py.File(self._filename, 'r') as hf:
            if group_name != "/":
                group = hf.get(group_name)
                ds = np.array(group.get(name=name))
            else:
                ds = np.array(hf.get(name=name))
        return ds

    @property
    def filename(self):
        return self._filename
