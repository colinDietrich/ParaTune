import numpy as np
from scipy.constants import c
from typing import List, Optional, Callable
from random import choices, randint
from abc import ABC, abstractmethod
import math
from ParaTune.media.data import *

class Crystal(ABC):
    """
    An abstract base class representing a nonlinear optical crystal.

    This class provides a template for creating specific types of nonlinear optical crystals with
    various domain configurations and properties. It initializes the crystal with essential parameters
    and provides methods to update crystal parameters, compute Fourier series coefficients, and reconstruct
    the domain structure from these coefficients.

    Attributes:
        configuration (str): The configuration of the crystal's design (e.g., 'normal', 'ppln', 'custom', 'random').
        medium (str): The type of medium (e.g., 'LiNbO3', 'KTP').
        number_grid_points_z (int): The number of grid points along the z-axis.
        wavelength_central (float): The pump central wavelength in meters.
        angular_frequency_central (float): The pump central angular frequency in radians per second, derived from the central wavelength.
        domain_width (Optional[float]): The domain width of each domain in meters. If not provided, it is calculated based on wavevector mismatch.
        length (Optional[float]): The crystal's length in meters. Required unless 'maximum_length' and 'minimum_length' are provided for random configuration.
        maximum_length (Optional[float]): The maximum possible length of the crystal in meters, used in random configurations.
        minimum_length (Optional[float]): The minimum possible length of the crystal in meters, used in random configurations.
        domain_values_custom (Optional[List[int]]): Custom orientation of the electric dipole for each domain, required if 'configuration' is 'custom'.
        domain_bounds_custom (Optional[List[float]]): Custom positions of the domain boundaries in meters, required if 'configuration' is 'custom'.
        data_medium: The physical properties of the crystal medium, determined by the 'medium' attribute.
        orientations (List[str]): Possible orientations of the crystal's electric field, dependent on the medium.
        number_of_domains (int): The number of domains in the crystal, calculated based on length and domain width.
        z_grid (np.ndarray): The grid array along the z-axis of the crystal.
        discretization_step_z (float): The discretization step size along the z-axis.
        domain_bounds (List[float]): The positions of the domain boundaries along the z-axis.
        domain_values (List[int]): The orientation of the electric field in each domain.
        fill (np.ndarray): An array used to construct the spatial representation of the domain structure.

    Methods:
        wavevector_mismatch(fundamental_angular_frequency, harmonic_angular_frequency): Abstract method to calculate the wavevector mismatch.
        update_crystal_parameters(): Updates the crystal parameters based on the current domain configuration.
        fourier_series_coeff_numpy(domain_values): Calculates the Fourier series coefficients for the domain structure.
        poling_expansion(fourrier_coefficients, period, z_grid): Reconstructs the domain structure from Fourier series coefficients.
        poling_function(domain_values): Constructs a spatial representation of the domain structure using domain values.
        sellmeier(A): Returns the refractive index as a function of frequency using the Sellmeier equation.
    """

    def __init__(self, 
                 configuration: str, 
                 medium: str, 
                 number_grid_points_z: int, 
                 domain_width: Optional[float] = None, 
                 length: Optional[float] = None,
                 maximum_length: Optional[float] = None, 
                 minimum_length: Optional[float] = None,
                 domain_values_custom: Optional[List[int]] = None,
                 domain_bounds_custom: Optional[List[float]] = None
                 ) -> None:
        
        # check for good initialization
        if(configuration == 'custom'):
            if(domain_values_custom is None and domain_bounds_custom is None):
                raise ValueError('parameters domain_values_custom and domain_bounds_custom missing for custom configuration.')
            if(length is None):
                raise ValueError('parameter length missing for custom configuration.')
            if(maximum_length is not None and minimum_length is not None):
                raise ValueError('Too much parameters for custom configuration.') 
        elif(configuration == 'normal' or configuration == 'ppln' or configuration == 'random'):
            if(domain_values_custom is not None and domain_bounds_custom is not None):
                raise ValueError('Too much parameters for normal configuration.') 
            if(length is None and (maximum_length is None or minimum_length is None)):
                raise ValueError('Length parameters missing for normal configuration.')

        # Parameters declaration
        self.configuration = configuration # configuration of the crystal's design -> normal / ppln / custom / random
        self.medium = medium  # type of medium -> LiNbO3 / KTP
        self.number_grid_points_z = number_grid_points_z # number of grid points along z [m]
        self.domain_width = domain_width # domain width of each domain [m]
        self.length = length # crystls's length [m]
        self.maximum_length = maximum_length # crystsl's maximum length [m]
        self.minimum_length = minimum_length # crystsl's minimum length [m]
        self.domain_values_custom = domain_values_custom # custom orientaion of the electric dipole of each domain
        self.domain_bounds_custom = domain_bounds_custom # custom positions of the boundaries of each domain [m]
        # variables for genetic algorithn
        self.signal_spectrum = None
        self.idler_spectrum = None
        self.level = 1
        
        # initialization of crystal's length
        if(length is not None):
            self.number_of_domains = (int)(self.length/self.domain_width) + 1
        elif(self.maximum_length is not None and self.minimum_length is not None):
            self.number_of_domains = randint(math.ceil(self.minimum_length/self.domain_width), math.floor(self.maximum_length/self.domain_width))
            self.length = self.number_of_domains*self.domain_width
        else:
            raise ValueError('Length of crystal missing.')
        
        # grid array along z axis
        self.z_grid = np.linspace(0, self.length, self.number_grid_points_z)
        # discretization step along z
        self.discretization_step_z = self.length/self.number_grid_points_z

        # configuration initialization
        if(self.configuration == 'normal'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.number_of_domains+1)]
            self.domain_values = [1,1]*(int(self.number_of_domains/2))
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.configuration == 'ppln'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.number_of_domains+1)]
            self.domain_values = [1,-1]*(int(self.number_of_domains/2))
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.configuration == 'random'):
            self.domain_bounds = [0 + self.domain_width*i for i in range(self.number_of_domains+1)]
            self.domain_values = choices([1, -1], k=self.number_of_domains)
            if(len(self.domain_values) < len(self.domain_bounds)): self.domain_values.append(1)
        elif(self.configuration == 'custom'):
            if(domain_values_custom is not None and domain_bounds_custom is not None):
                self.number_of_domains = len(domain_values_custom)
                self.domain_values = domain_values_custom
                self.domain_bounds = domain_bounds_custom
            else:
                raise ValueError('Custom domain values and bound are not given for crystal configuration.')
        else:
            raise ValueError('Unkown configuration for the crystal.')
        
        # Check that the discretization step is at least smaller than the half width of a domain
        print(f"domain_width/2 = {self.domain_width/2} ")
        print(f"dictretization step along z = {self.discretization_step_z}")
        if(self.discretization_step_z > self.domain_width/2):
            print('Discretization step along z could be too big. Increase the number of grid points along z such that it is at least twice the width of domain')

        self.fill = np.ones((self.domain_width/(self.z_grid[1]-self.z_grid[0])+1).astype(int))


    @abstractmethod
    def wavevector_mismatch(self):
        pass

    def update(self, level=1) -> None:
        if(level != self.level):
            self.level = level
            self.domain_width = self.domain_width/2
            self.domain_values = self.double_length_array(self.domain_values)
        self.number_of_domains = len(self.domain_values)
        # length of crystal
        self.length = self.number_of_domains*self.domain_width
        # array of positions of each domain
        self.z_grid = np.arange(-self.length/2, -self.length/2 + (self.number_of_domains + 1) * self.domain_width, self.domain_width)
        self.number_grid_points_z = len(self.z_grid)

    def double_length_array(self, array) -> np.ndarray:
        doubled_array = []
        for element in array:
            doubled_array.extend([element, element])
        return doubled_array

    def update_crystal_parameters(self) -> None:
        """
        Updates the crystal parameters based on the current domain configuration.

        This method recalculates the crystal's length, step size along the z-axis, domain boundaries,
        and initializes an array to represent the domain structure along the z-axis based on the current
        domain values and width.

        It is typically called after any modification to the domain structure to ensure the crystal's
        properties reflect the updated configuration.
        """
        self.number_of_domains = len(self.domain_values)
        self.length = self.number_of_domains*self.domain_width
        self.discretization_step_z = self.length/self.n
        self.domain_bounds = [0 + self.domain_width*i for i in range(self.number_of_domains+1)]
        self.z_grid = np.linspace(0,self.length,self.number_grid_points_z)
        self.fill = np.ones((self.domain_width/(self.z_grid[1]-self.z_grid[0])+1).astype(int))


    def fourier_series_coeff_numpy(self, domain_values: List[int]) -> np.ndarray:
        """
        Calculates the Fourier series coefficients for the domain structure of a crystal, represented as a periodic function. 
        This computation is crucial for analyzing the crystal's domain structure and its implications on the efficiency of 
        nonlinear optical interactions.

        The Fast Fourier Transform (FFT) is utilized to compute these coefficients, providing a quick and efficient way to 
        understand the frequency components within the crystal's domain structure. The coefficients can be used to analyze 
        and interpret the periodicity and symmetry properties of the domain structure, which are essential for optimizing 
        the crystal's optical properties.

        Parameters:
            domain_values (List[int]): A list of integers representing the domain orientation values within the crystal. 
                                    These values indicate the direction of the electric field in each domain, typically 
                                    encoded as positive and negative values corresponding to the orientation of the 
                                    ferroelectric domains.

        Returns:
            np.ndarray: An array of complex numbers representing the Fourier series coefficients of the crystal's domain 
                        structure. Each coefficient corresponds to a specific frequency component, with its magnitude 
                        indicating the component's contribution to the overall domain structure and its phase representing 
                        the component's alignment within the periodic function.

        Note:
            The Fourier series representation of a periodic function f(t) is given by the sum of sinusoidal functions:
                f(t) ≈ a0/2 + Σ (a_k * cos(2πkt/T) + b_k * sin(2πkt/T)) for k = 1 to N,
            where T is the period, and N is the number of coefficients to compute. This method simplifies the representation 
            by using the FFT, which inherently includes both cosine and sine components in its complex output.
        """
        # Convert the list of domain values to a NumPy array for efficient computation
        domain_values_array = np.array(domain_values)
        
        # Compute the Fast Fourier Transform (FFT) of the domain values
        # The FFT result is a complex array where each value represents a frequency component
        fft_result = np.fft.fft(domain_values_array)
        
        # Normalize the FFT result to get the coefficients
        # The normalization factor is usually the length of the domain values array
        # This step converts the FFT result into the actual Fourier series coefficients
        coefficients = fft_result / len(domain_values_array)
        
        return coefficients


    def poling_expansion(self, fourrier_coefficients: np.ndarray) -> np.ndarray:
        """
        Approximates the original domain values from Fourier series coefficients using the Inverse Fast Fourier Transform (IFFT). 
        This function is particularly useful in reconstructing the spatial domain structure of a crystal from its frequency 
        domain representation, which is obtained from the Fourier coefficients.

        The IFFT process converts the frequency domain information back into the spatial domain, providing an approximation of 
        the original domain orientations. This approximation is especially relevant in the context of ferroelectric domain 
        engineering in nonlinear optical crystals, where the precise arrangement of domains affects the crystal's optical properties.

        Parameters:
            fourier_coefficients (np.ndarray): An array of Fourier series coefficients obtained from the FFT of the crystal's 
                                            domain structure. These coefficients include both magnitude and phase information 
                                            essential for the reconstruction.

        Returns:
            np.ndarray: An array of real numbers representing the approximated original domain values normalized to the maximum 
                        value. This normalization is typically performed to facilitate the comparison and analysis of the 
                        reconstructed domain structure with respect to its original form.

        Note:
            The accuracy of the reconstruction depends on the completeness and accuracy of the Fourier coefficients. Since the 
            IFFT uses both magnitude and phase information, any loss or approximation in these values during the FFT process 
            can affect the fidelity of the reconstructed domain values.
        """
        # Compute the IFFT
        domain_values_approx = np.fft.ifft(fourrier_coefficients)

        # Since the original domain values are real, take the real part of the IFFT result
        return np.real(domain_values_approx)/np.max(np.real(domain_values_approx))

    def poling_function(self, domain_values: List[int]) -> np.ndarray:
        """
        Constructs a spatial representation of the crystal's domain structure using the provided domain values.

        This method applies a vectorized mapping to extend each domain value across its corresponding domain
        width as defined by the 'fill' attribute, effectively creating a piecewise constant function that
        represents the orientation of the electric field in each domain along the crystal's z-axis.

        Parameters:
            domain_values (List[int]): A list of integers representing the orientation of the electric field
                                    in each domain. Typically, values are +1 or -1, indicating the direction
                                    of poling in each domain.

        Returns:
            np.ndarray: A flattened numpy array representing the spatial domain structure of the crystal along
                        the z-axis, with each domain's electric field orientation extended across its width.
        """
        extended_domains = []
        for value in domain_values:
            extended_domains.append(value * self.fill)
        return np.concatenate(extended_domains).flatten()

    def sellmeier(self, A: List[float]) -> Callable[[float], float]:
        """
        Returns the refractive index as a function of frequency using the Sellmeier equation.

        Parameters:
            A (List[float]): List of Sellmeier coefficients.

        Returns:
            Callable[[float], float]: Function that calculates the refractive index for a given frequency.
        """
        return lambda x: np.sqrt(A[0] + A[1]/((2*np.pi*c/x*1e6)**2 - A[2]) - A[3] * (2*np.pi*c/x*1e6)**2)
    
    def phase_matching_function(self, wave_number_span: float, wave_number_grid_points: int):
        """
        Calculates the phase matching function (PMF) for the crystal over a specified range of wave numbers.

        The PMF is a crucial factor in nonlinear optical processes, indicating how well the phases of interacting
        waves are matched over the length of the crystal. It is defined by the integral of the crystal's
        nonlinear coefficient profile, modulated by the exponential of the negative product of the wave number 
        mismatch and the position within the crystal.

        Parameters:
            wave_number_span (float): The range of wave numbers around the central mismatch to evaluate the PMF.
            wave_number_grid_points (int): The number of points to discretize the wave number span for PMF calculation.

        Returns:
            wave_number_array (np.ndarray): An array of wave numbers over which the PMF is evaluated.
            pmf (np.ndarray): The calculated phase matching function values corresponding to the wave_number_array.

        Note:
            The PMF is calculated for each domain of the crystal, and the contributions from all domains are
            summed to obtain the total PMF. This function assumes a piecewise constant nonlinear coefficient profile
            across the crystal's domains.
        """
        # Calculate the central phase mismatch based on the crystal's central angular frequency
        wavevector_mismatch_central = self.wavevector_mismatch()(self.angular_frequency_central/2, self.angular_frequency_central/2)

        # Generate an array of wave numbers around the central mismatch
        wave_number_array = np.linspace(wavevector_mismatch_central - wave_number_span / 2,
                                        wavevector_mismatch_central + wave_number_span / 2,
                                        wave_number_grid_points)

        # Define a lambda function to calculate the PMF contribution from a single domain
        pmf_one_domain = lambda z1, z2: (1 / self.length) * 1j * \
                        (np.exp(1j * wave_number_array * z1) - np.exp(1j * wave_number_array * z2)) / wave_number_array

        # Initialize the total PMF to zero
        pmf = 0

        # Convert the domain values to a NumPy array for processing
        parameters = np.array(self.domain_values)

        # Loop over each domain to calculate and sum its contribution to the total PMF
        for idz in range(len(parameters) - 1):  # Subtract 1 to avoid index out of range since we're accessing idz+1
            pmf += parameters[idz] * pmf_one_domain(self.domain_bounds[idz], self.domain_bounds[idz + 1])

        # Return the array of wave numbers and the corresponding PMF values
        return wave_number_array, pmf