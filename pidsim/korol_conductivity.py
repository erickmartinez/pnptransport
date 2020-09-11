import numpy as np
from pidsim.conductivity_interface import ConductivityInterface


class KorolConductivity(ConductivityInterface):
    """
    This class provides methods to map a concentration of Na atoms in Si to a conductivity value.

    *Example*

    .. code-block:: python

        from pidsim.korol_conductivity import KorolConductivity
        import h5py

        conductivity_model: KorolConductivity = KorolConductivity()
        # Assume every Na atom contributes 1 conduction electron
        conductivity_model.activated_na_fraction = 1.
        # Get a concentration profile from a transport simulation
        h5_path = './transport_simulation_output.h5'
        # Get the profile at index 20
        idx = 20
        with h5py.File(h5_path, 'r') as hf:
            c = np.array(hf['/L2/concentration/ct_{0:d}'.format(idx)])
        # Update the concentration profile in the model
        conductivity_model.concentration_profile = c
        conductivity_model.segregation_coefficient = 1.
        conductivity = conductivity_model.estimate_conductivity()

    Attributes
    ----------
    __sodium_profile: np.ndarray
        The Na concentration profile in cm\ :sup:`-3` \.
    __activated_na_fraction: float
        The activated fraction of Na atoms to compute the conductivity :math:`0 < f < 1`.
    __segregation_coefficient: float
        .. deprecated:: 0.1
            This value represents the segregation coefficient of Na in the stacking fault assuming a mechanism driven
            by bulk diffusion + segregation at the SF. Use 1.0.

    """
    __sodium_profile: np.ndarray = None
    __activated_na_fraction: float = 1.
    __segregation_coefficient: float = 50

    def estimate_conductivity(self):
        if self.__sodium_profile is None:
            raise ValueError('Set the sodium concentration profile before trying to estimate the conductivity.')
        return self.conductivity_model(self.__sodium_profile)

    @property
    def concentration_profile(self):
        return self.__sodium_profile

    @concentration_profile.setter
    def concentration_profile(self, value):
        self.__sodium_profile = np.abs(value)

    @property
    def activated_na_fraction(self) -> float:
        return self.__activated_na_fraction

    @activated_na_fraction.setter
    def activated_na_fraction(self, value: float):
        self.__activated_na_fraction = abs(value)

    @property
    def segregation_coefficient(self) -> float:
        return self.__segregation_coefficient

    @segregation_coefficient.setter
    def segregation_coefficient(self, value):
        self.__segregation_coefficient = np.abs(value)

    def conductivity_model(self, concentration: np.ndarray) -> np.ndarray:
        """
        Implementation of the conductivity_model model.


        **Model simplifications**

        1. The Na to Si ratio in the stacking fault is obtained from the ratio between Na concentration and Si
           concentration in the bulk of a perfect crystal (does not consider the specific geometry of a stacking fault)
        2. Conductivity is calculated based on depth-resolved Hall-effect measurements of mobility and carrier density
           in Na-implanted Si (Korol et al.)

        *Reference*
            Korol, V. M. "Sodium ion implantation into silicon." Physica status solidi (a) 110.1 (1988): 9-34.

        Parameters
        ----------
        concentration: np.ndarray
            The sodium concentration in the Si bulk

        Returns
        -------
        np.ndarray:
            The conductivity_model profile
        """

        # Na concentration in the shunt
        cshunt = concentration * self.__segregation_coefficient * self.__activated_na_fraction

        # Model based on implantation data
        # Korol, V. M. "Sodium ion implantation into silicon." Physica status solidi (a) 110.1 (1988): 9-34.
        # Fitting of coefficients in Extract_NaImp.py
        coord = -11.144769029961262
        slope = 0.717839509854622

        sigma = np.power(10, coord) * np.power(cshunt, slope)  # (10 ** coord) * (cshunt ** slope)  # S/cm

        return sigma
