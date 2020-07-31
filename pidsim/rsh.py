import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import h5py

class Rsh:
    _shunt_width: float = None
    _shunt_height: float = 