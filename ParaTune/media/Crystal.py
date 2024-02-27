import jax.numpy as np
from functools import partial
import jax
from scipy.constants import c
from random import choices, randint
from typing import Union, Callable, List, Optional

# for Sellmeier coefficient, see : https://www.unitedcrystals.com/NLOCOverview.html
# for second-order suceptibility coefficients, see : https://www.sciencedirect.com/topics/chemistry/second-order-nonlinear-optical-susceptibility

# nb : The second and third indices j,k of d_ijk are then replaced by a single symbol l according to the piezoelectric contraction :
# jk:	11	22	33	23,32	31,13	12,21
# l:	1	2	3	4	    5	    6

# Lithium Niobate
LN = {
    "name": 'LN',
    "x": [4.9048, 0.11768, 0.04750, 0.027169], # wavelength in 1e-6 m
    "y": [4.5820, 0.099169, 0.04443, 0.021950], # wavelength in 1e-6 m
    "d31": 7.11*1e-12, # type 1
    "d22": 3.07*1e-12, # type 0
    "d33": 29.1*1e-12, # type 0
}


# Potassium Titanyl Phosphate Single Crytal
KTP = {
    "name": 'KTP',
    "x": [3.0065, 0.03901, 0.04251, 0.01327], # wavelength in 1e-6 m
    "y": [3.0333, 0.04154, 0.04547, 0.01408], # wavelength in 1e-6 m
    "z": [3.3134, 0.05694, 0.05658, 0.01682], # wavelength in 1e-6 m
    "d24": 3.64*1e-12,  # type 2
    "d31": 2.54*1e-12,  # type 1
    "d32": 4.35*1e-12,  # type 1
    "d33": 16.9*1e-12,  # type 0
}

class Crystal:
    """
    Represents a nonlinear optical crystal with configurable properties for
    simulating nonlinear optical interactions such as second-harmonic generation (SHG) or
    spontaneous parametric down-conversion (SPDC).

    Attributes:
        config (str): Configuration of the crystal's nonlinear domain structure.
        medium (str): Type of nonlinear optical medium (e.g., 'LiNbO3', 'KTP').
        n (int): Number of grid points along the propagation direction (z-axis).
        interaction (str): Type of nonlinear optical interaction ('shg' for SHG, 'spdc' for SPDC).
        wl_central (float): Central wavelength of the optical interaction.
        domain_width (Optional[float]): Poling period of the nonlinear domain structure.
        length (Optional[float]): Length of the crystal.
        max_length (Optional[float]): Maximum length of the crystal for random configuration.
        min_length (Optional[float]): Minimum length of the crystal for random configuration.
        domain_values_custom (Optional[List[int]]): Custom domain orientation values for custom configuration.
        domain_bounds_custom (Optional[List[float]]): Custom domain boundary positions for custom configuration.
        signal (Optional[str]): Orientation of the signal wave (used in 'spdc').
        idler (Optional[str]): Orientation of the idler wave (used in 'spdc').
        pump (Optional[str]): Orientation of the pump wave (used in 'spdc').
        fundamental (Optional[str]): Orientation of the fundamental wave (used in 'shg').
        second_harmonic (Optional[str]): Orientation of the second harmonic wave (used in 'shg').
    """
    def __init__(self, 
                config: str, 
                medium: str,
                n: int, 
                interaction: str,
                wl_central: float,
                domain_width: Optional[float] = None, 
                length: Optional[float] = None,
                max_length: Optional[float] = None, 
                min_length: Optional[float] = None,
                domain_values_custom: Optional[List[int]] = None,
                domain_bounds_custom: Optional[List[float]] = None,
                signal: Optional[str] = None, 
                idler: Optional[str] = None, 
                pump: Optional[str] = None,
                fundamental: Optional[str] = None, 
                second_harmonic: Optional[str] = None
                ) -> None:

        self.config = config    # configuration of the crystal's design -> normal / ppln / custom
        self.medium = medium    # type of medium -> LiNbO3 / KTP
        self.n = n    # number of grid points along z
        self.interaction = interaction
        self.wl_central = wl_central
        self.freq_central = 2*np.pi*c/wl_central
        self.z = None   # grid along direction of propagation
        self.data_medium = None     # data about the medium
        self.domain_width = None  # domain width of each domain
        self.domain_bounds = None    # positions of the boundaries of each domain
        self.domain_values = None     # orientaion of the electric dipole of each domain
        self.domain_expansion = None    # Fourier expansion of domain_values
        self.wavevector_mismatch = None  # wavevector mismatch function -> shg / spdc
        self.d = None     # bulk nonlinear coefficient
        self.deff = None  # effective nonlinear coefficient
        self.k_f = None     # wave number as function of frequency in [rad/s] along fundamental axis
        self.k_sh = None
        self.n_f = None     # refraction index as function of frequency in [rad/s] along fundamental axis
        self.n_sh = None
        self.k_s = None
        self.k_i = None
        self.k_p = None
        self.n_s = None
        self.n_i = None
        self.n_p = None
        self.orientations = []
        self.N = None # number of domains
        self.fitness = None
        self.out = None
        self.length = None
        self.step = None  # discretization step along z

        if(medium == 'LiNbO3'):
            self.data_medium = LN
            self.orientations = ['x', 'y']
        elif(medium == 'KTP'):
            self.data_medium = KTP
            self.orientations = ['x', 'y', 'z']
        else:
            raise ValueError('Unkown medium for the crystal.')

        if(domain_width is not None):
            self.domain_width = domain_width

        if(interaction == 'shg'):
            if(fundamental is not None and second_harmonic):
                if(fundamental in self.orientations and second_harmonic in self.orientations):
                    self.n_f = self.sellmeier(self.data_medium[fundamental])
                    self.n_sh = self.sellmeier(self.data_medium[second_harmonic])
                    self.k_f = lambda freq: freq / c * self.n_f(freq)
                    self.k_sh = lambda freq: freq / c * self.n_sh(freq)
                    self.d = self.data_medium["d31"]
                    self.deff = (2/np.pi) * self.d
                else:
                    raise ValueError('Incorrect value for fundamental/second harmonic orientations.')
            else:
                raise ValueError('Not enough values for fundamental/second harmonic orientations.')
            self.wavevector_mismatch = self.wavevector_mismatch_shg()
            dk = self.wavevector_mismatch(self.freq_central, self.freq_central, self.freq_central*2)
            if(self.domain_width is None): self.domain_width = np.abs(np.pi / dk)
        elif(interaction == 'spdc'):
            if(signal is not None and idler is not None and pump is not None):
                if(signal in self.orientations and idler in self.orientations and pump in self.orientations):
                    self.n_s = self.sellmeier(self.data_medium[signal])
                    self.n_i = self.sellmeier(self.data_medium[idler])
                    self.n_p = self.sellmeier(self.data_medium[pump])
                    self.k_s = lambda freq: freq / c * self.n_s(freq)
                    self.k_i = lambda freq: freq / c * self.n_i(freq)
                    self.k_p = lambda freq: freq / c * self.n_p(freq)
                    self.d = self.data_medium["d31"]
                    self.deff = (2/np.pi) * self.d
                else:
                    raise ValueError('Incorrect value for signal/idler/pump orientation.')
            else:
                raise ValueError('Not enough values for fundamental/second harmonic orientations.')
            self.wavevector_mismatch = self.wavevector_mismatch_spdc()
            dk = self.wavevector_mismatch(self.freq_central/2, self.freq_central/2)
            if(self.domain_width is None): self.domain_width = np.abs(np.pi / dk)
        else:
                raise ValueError('Unkown interaction for the crystal. Must be shg or spdc.')

        if(length is not None):
            self.length = length  # length of crystal
            self.step = self.length/n
        elif(max_length is not None and min_length is not None):
            self.N = randint(jax.lax.ceil(min_length/self.domain_width), jax.lax.floor(max_length/self.domain_width))
            self.length = self.N*self.domain_width
            self.step = self.length/n
        else:
            raise ValueError('Length of crystal missing.')

        self.z = np.linspace(0,self.length,self.n)

        if(self.N is None):
            self.N = (int)(self.length/self.domain_width) + 1

        print("domain_width/2 = " + str(self.domain_width/2))
        print("domain_width/5 = " + str(self.domain_width/5))
        print("dictretization step along z = " + str(self.step))
        if(self.step > self.domain_width/5):
          print('Discretization step along z could be too big. Increase the number of grid points along z.')

        if(self.config == 'normal'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.N+1)]
            self.domain_values = [1,1]*(int(self.N/2))
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.config == 'ppln'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.N+1)]
            self.domain_values = [1,-1]*(int(self.N/2))
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.config == 'random'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.N+1)]
            self.domain_values = choices([1, -1], k=self.N)
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.config == 'custom'):
            if(domain_values_custom is not None and domain_bounds_custom is not None):
                self.N = len(domain_values_custom)
                self.domain_values = domain_values_custom
                self.domain_bounds = domain_bounds_custom
            else:
                raise ValueError('Custom domain values and bound are not given for crystal configuration.')
        else:
            raise ValueError('Unkown configuration for the crystal.')

        N_domains_alongs_z = (self.domain_width/(self.z[1]-self.z[0])+1).astype(int)
        self.fill = np.ones(N_domains_alongs_z)

    def update_parameters(self) -> None:
        """
        Updates the crystal parameters based on the current domain configuration.

        This method recalculates the crystal's length, step size along the z-axis, domain boundaries,
        and initializes an array to represent the domain structure along the z-axis based on the current
        domain values and width.

        It is typically called after any modification to the domain structure to ensure the crystal's
        properties reflect the updated configuration.
        """
        self.N = len(self.domain_values)
        self.length = self.N*self.domain_width
        self.step = self.length/self.n
        self.domain_bounds = [0 + self.domain_width*i for i in range(self.N+1)]
        self.z = np.linspace(0,self.length,self.n)
        N_domains_alongs_z = (self.domain_width/(self.z[1]-self.z[0])+1).astype(int)
        self.fill = np.ones(N_domains_alongs_z)


    def fourier_series_coeff_numpy(self, domain_values: List[int]) -> np.ndarray:
        """
        Calculates the Fourier series coefficients for the domain structure of the crystal represented as a periodic function.

        The method uses the Fast Fourier Transform (FFT) to compute the coefficients, providing insight into the
        frequency components of the domain structure, which is essential for analyzing the efficiency of nonlinear
        optical interactions within the crystal.

        Parameters:
            domain_values (List[int]): A list of domain orientation values, indicating the direction of the
                                    electric field in each domain.

        Returns:
            np.ndarray: An array of Fourier series coefficients representing the frequency components of the
                        crystal's domain structure.

        Note:
            The Fourier series representation is given by:
            f(t) ~= a0/2 + sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )
            where f(t) is the periodic function representing the domain structure, T is the period,
            and N is the number of coefficients to compute.
    """

        arr = np.array(self.poling_function(domain_values))
        y = np.fft.fft(arr) / len(arr)
        return y


    def poling_expansion(self, y: np.ndarray, period: float, z: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Reconstructs the spatial function of the crystal's domain structure from its Fourier series coefficients.

        This method applies the inverse process of a Fourier transform to reconstruct the domain structure
        as a function of position along the crystal's z-axis, using the computed Fourier series coefficients.

        Parameters:
            y (ndarray): The Fourier series coefficients of the domain structure, obtained from a Fourier transform.
            period (float): The period of the domain structure, equivalent to the poling period in the case of
                            periodically poled materials.
            z (Union[List[float], ndarray]): The spatial grid along the z-axis of the crystal over which the
                                            domain structure is to be reconstructed.

        Returns:
            ndarray: A real-valued array representing the reconstructed domain structure along the crystal's z-axis.

        Note:
            The domain structure is approximated by the Fourier series:
            f(z) ~= a0/2 + sum_{k=1}^{N} ( a_k*cos(2*pi*k*z/period) + b_k*sin(2*pi*k*z/period) )
            where a0 is the zeroth Fourier coefficient, a_k are the real parts of the subsequent coefficients,
            b_k are the imaginary parts, and N is the number of coefficients used in the reconstruction.
        """
        a0 = y[0].real/2
        a_k = y[1:-1].real
        b_k = y[1:-1].imag
        pol = a0
        z = np.array(z)
        for k in range(len(a_k)):
            pol = pol + a_k[k] * np.cos(2*np.pi*k*z/period) - b_k[k] * np.sin(2*np.pi*k*z/period)

        return pol.real

    @partial(jax.jit, static_argnums=(0,))
    def poling_function(self, domain_values: List[int]) -> np.ndarray:
        """
        Constructs a spatial representation of the crystal's domain structure using the provided domain values.

        This method applies a vectorized mapping to extend each domain value across its corresponding domain
        width as defined by the 'fill' attribute, effectively creating a piecewise constant function that
        represents the orientation of the electric field in each domain along the crystal's z-axis.

        The JAX `jit` decorator is used to just-in-time compile the function for faster execution, and `vmap`
        is utilized to efficiently apply the 'extend' operation across all domain values.

        Parameters:
            domain_values (List[int]): A list of integers representing the orientation of the electric field
                                    in each domain. Typically, values are +1 or -1, indicating the direction
                                    of poling in each domain.

        Returns:
            np.ndarray: A flattened numpy array representing the spatial domain structure of the crystal along
                        the z-axis, with each domain's electric field orientation extended across its width.
        """
        extend = lambda x : x*self.fill
        extend_vmap = jax.vmap(extend)
        return extend_vmap(domain_values).flatten()

    def sellmeier(self, A: List[float]) -> Callable[[float], float]:
        """
        Returns the refractive index as a function of frequency using the Sellmeier equation.

        Parameters:
            A (List[float]): List of Sellmeier coefficients.

        Returns:
            Callable[[float], float]: Function that calculates the refractive index for a given frequency.
        """
        return lambda x: np.sqrt(A[0] + A[1]/((2*np.pi*c/x*1e6)**2 - A[2]) - A[3] * (2*np.pi*c/x*1e6)**2)

    def wavevector_mismatch_shg(self) -> Callable[[float, float, float], float]:
        """
        Returns a function to calculate the wavevector mismatch for second-harmonic generation.

        Returns:
            Callable[[float, float, float], float]: Function that calculates the wavevector mismatch given the frequencies of the fundamental and second-harmonic waves.
        """
        return lambda wf1, wf2, wsh:  self.k_f(wf1) + self.k_f(wf2) - self.k_sh(wsh)

    def wavevector_mismatch_spdc(self) -> Callable[[float, float], float]:
        """
        Returns a function to calculate the wavevector mismatch for spontaneous parametric down-conversion.

        Returns:
            Callable[[float, float], float]: Function that calculates the wavevector mismatch given the frequencies of the signal and idler waves.
        """
        return lambda ws, wi: self.k_s(ws)+self.k_i(wi)-self.k_p(ws+wi)