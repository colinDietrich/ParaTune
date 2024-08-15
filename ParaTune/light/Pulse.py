from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import h, c
from typing import Callable
from scipy.interpolate import interp1d

class Pulse(ABC):
    """
    Abstract base class representing a pulse of light, characterized by its spectral and temporal properties.

    Attributes:
        wavelength_central (float): Central wavelength of the pulse in meters.
        wavelength_bandwidth (float): Spectral bandwidth of the pulse in meters.
        mean_power (float): Average power of the pulse in Watts.
        repetition_rate (float): Repetition rate of the pulse in Hertz.
        number_of_grid_points (int): Number of grid points for numerical simulations.
        wavelength_span (float): Total span of wavelengths to consider in meters.
        refractive_index_function (Callable[[float], float]): Function to calculate the refractive index as a function of angular frequency.
    """

    def __init__(self,
                 wavelength_central: float,
                 wavelength_bandwidth: float,
                 mean_power: float,
                 repetition_rate: float, 
                 number_of_grid_points: int, 
                 wavelength_span: float,
                 refractive_index_function: Callable[[float], float]
                 ):
        """
        Initializes the Pulse object with the given parameters.

        Parameters:
            wavelength_central (float): Central wavelength of the pulse in meters.
            wavelength_bandwidth (float): Spectral bandwidth of the pulse in meters (standard deviation).
            mean_power (float): Average power of the pulse in Watts.
            repetition_rate (float): Repetition rate of the pulse in Hertz.
            number_of_grid_points (int): Number of grid points for numerical simulations.
            wavelength_span (float): Total span of wavelengths to consider in meters.
            refractive_index_function (Callable[[float], float]): Function to calculate the refractive index as a function of angular frequency.
        """
        
        assert number_of_grid_points > 0, "Number of grid points must be positive."
        self.number_of_grid_points = number_of_grid_points  # number of grid points

        # wavelength parameters
        self._wavelength_central = wavelength_central  # central wavelength [m]
        self._wavelength_span = wavelength_span # wavelength window [m]
        self._wavelength_bandwidth = wavelength_bandwidth # wavelength bandwidth of the pulse [m]
        self._wavelength_grid = np.linspace(self._wavelength_central - self.wavelength_span/2,
                                           self._wavelength_central + self.wavelength_span/2, 
                                           self.number_of_grid_points) # wavelength grid [m]
        self._mean_power = mean_power # mean power [W]
        self._repetition_rate = repetition_rate # repetition rate [Hz]

        # refractive index of the media in which the pulse propagate
        self.refractive_index_function = refractive_index_function

        # frequency parameters
        self._frequency_central = self.convert_wavelength_to_frequency(self.wavelength_central)  # central frequency [Hz]
        self._frequency_bandwidth = self.get_frequency_bandwidth() # spectral bandwidth of the pulse [Hz]
        self._frequency_grid, self._frequency_amplitude = self.wavelength_to_frequency_grid(
                                                                self.wavelength_grid, 
                                                                self.wavelength_amplitude) # frequency grid / amplitude
        
        # temporal parameters
        self._time_grid, self._time_amplitude = self.get_time_amplitude() # time grid / amplitude

    @property
    @abstractmethod
    def wavelength_amplitude(self) -> float:
        """
        Abstract property that should return the amplitude of the pulse as a function of wavelength.
        Must be implemented by subclasses.
        """
        pass

    @property
    def wavelength_central(self) -> float:
        return self._wavelength_central  
    
    @property
    def wavelength_span(self) -> float:
        return self._wavelength_span
    
    @property
    def wavelength_bandwidth(self) -> float:
        return self._wavelength_bandwidth
    
    @property
    def wavelength_grid(self) -> np.ndarray[float]:
        return self._wavelength_grid
    
    @property
    def mean_power(self) -> float:
        return self._mean_power
    
    @property
    def repetition_rate(self) -> float:
        return self._repetition_rate
    
    @property
    def energy_per_photon(self) -> float:
        return h * c / self.wavelength_central
    
    @property
    def energy_per_pulse(self) -> float:
        return self.mean_power / self.repetition_rate
    
    @property 
    def average_photons_per_pulse(self) -> float:
        return self.energy_per_pulse / self.energy_per_photon
    
    @property
    def photons_per_second(self) -> float:
        return self.average_photons_per_pulse * self._repetition_rate
    
    @property
    def frequency_central(self) -> float:
        return self._frequency_central
    
    @property 
    def frequency_bandwidth(self) -> float:
        return self._frequency_bandwidth
    
    @property
    def frequency_grid(self) -> np.ndarray[complex]:
        return self._frequency_grid
    
    @property
    def frequency_amplitude(self) -> np.ndarray[complex]:
        return self._frequency_amplitude
    
    @property
    def time_grid(self) -> np.ndarray[float]:
        return self._time_grid
    
    @property
    def time_amplitude(self) -> np.ndarray[complex]:
        return self._time_amplitude

    @wavelength_central.setter
    def wavelength_central(self, new_wavelength_central: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._wavelength_central = new_wavelength_central
    
    @wavelength_span.setter
    def wavelength_span(self, new_wavelength_span: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._wavelength_span = new_wavelength_span
    
    @wavelength_bandwidth.setter
    def wavelength_bandwidth(self, new_wavelength_bandwidthl: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._wavelength_bandwidth = new_wavelength_bandwidthl
    
    @wavelength_grid.setter
    def wavelength_grid(self, new_wavelength_grid: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._wavelength_grid = new_wavelength_grid
    
    @mean_power.setter
    def mean_power(self, new_mean_power: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._mean_power = new_mean_power
    
    @repetition_rate.setter
    def repetition_rate(self, new_repetition_rate: float) -> None:
        # update frequency and time parameters
        self.update_frequency_and_time_parameters()
        self._repetition_rate = new_repetition_rate

    @frequency_central.setter
    def frequency_central(self, new_frequency_central: float) -> None:
        # update frequency and time parameters
        self.update_wavelength_and_time_parameters()
        self._frequency_central = new_frequency_central
    
    @frequency_bandwidth.setter 
    def frequency_bandwidth(self, new_frequency_bandwidth: float) -> None:
        # update frequency and time parameters
        self.update_wavelength_and_time_parameters()
        self._frequency_bandwidth = new_frequency_bandwidth
    
    @frequency_grid.setter
    def frequency_grid(self, new_frequency_grid: np.ndarray[float]) -> None:
        # update frequency and time parameters
        self.update_wavelength_and_time_parameters()
        self._frequency_grid = new_frequency_grid
    
    @frequency_amplitude.setter
    def frequency_amplitude(self, new_frequency_amplitude: np.ndarray[float]) -> None:
        # update frequency and time parameters
        self.update_wavelength_and_time_parameters()
        self._frequency_amplitude = new_frequency_amplitude
    
    def update_frequency_and_time_parameters(self) -> None:
        # frequency parameters
        self._frequency_central = self.convert_wavelength_to_frequency(self.wavelength_central)  # central frequency [Hz]
        self._frequency_bandwidth = self.get_frequency_bandwidth() # spectral bandwidth of the pulse [Hz]
        self._frequency_grid, self._frequency_amplitude = self.wavelength_to_frequency_grid(
                                                                self.wavelength_grid, 
                                                                self.wavelength_amplitude) # frequency grid / amplitude
        # temporal parameters
        self._time_grid, self._time_amplitude = self.get_time_amplitude() # time grid / amplitude

    def update_wavelength_and_time_parameters(self) -> None:
        # wavelength parameters
        self._wavelength_grid, self._wavelength_amplitude = self.frequency_to_wavelength_grid(
                                                                self.frequency_grid, 
                                                                self.frequency_amplitude) # frequency grid / amplitude
        # temporal parameters
        self._time_grid, self._time_amplitude = self.get_time_amplitude() # time grid / amplitude

    def convert_wavelength_to_frequency(self, wavelength: float) -> float:
        """
        Converts a given wavelength to frequency using the light speed in vacuum and the refractive index.

        Parameters:
            wavelength (float): Wavelength in meters.

        Returns:
            float: Corresponding frequency in Hertz.
        """
        return c / (self.refractive_index_function(2*np.pi*c/wavelength) * wavelength)
    
    def convert_frequency_to_wavelength(self, frequency: float) -> float:
        """
        Converts a given frequency to its corresponding wavelength in a medium,
        taking into account the medium's refractive index.

        Parameters:
            frequency (float): The frequency in Hertz to be converted.

        Returns:
            float: The corresponding wavelength in meters.
        """
        return c / (self.refractive_index_function(2*np.pi*frequency) * frequency)
    
    def get_frequency_bandwidth(self) -> float:
        """
        Calculates the spectral bandwidth of the pulse in the frequency domain,
        based on its wavelength bandwidth and the refractive index function.

        Returns:
            float: The frequency bandwidth in Hertz.
        """
        # Numerically approximate the derivative of the refractive index function
        delta_lambda = 1e-9  # small change in wavelength, in meters
        n_lambda = self.refractive_index_function(2*np.pi*c/self.wavelength_central)
        n_lambda_delta = self.refractive_index_function(2*np.pi*c/self.wavelength_central + delta_lambda)
        dn_dlambda = (n_lambda_delta - n_lambda) / delta_lambda

        # Calculate the frequency bandwidth using the given formula
        return abs(-c / self.wavelength_central**2 * 
                                  1 / (n_lambda + self.wavelength_central * dn_dlambda) * 
                                  self.wavelength_bandwidth)
    
    def wavelength_to_frequency_grid(self, 
                                     wavelength_grid: np.ndarray[float], 
                                     wavelength_amplitude: np.ndarray[float], 
                                     ) -> tuple[np.ndarray[float], np.ndarray[complex]]:
        """
        Converts a grid of wavelengths and their corresponding amplitudes to a frequency grid and amplitudes,
        accounting for the refractive index.

        Parameters:
            wavelength_grid (np.ndarray[float]): Array of wavelength values.
            wavelength_amplitude (np.ndarray[float]): Array of corresponding amplitude values.

        Returns:
            tuple[np.ndarray[float], np.ndarray[complex]]: The frequency grid and corresponding amplitude values.
        """
        frequency_grid = self.convert_wavelength_to_frequency(wavelength_grid)
        frequency_amplitude = wavelength_amplitude
        f = interp1d(frequency_grid, frequency_amplitude, fill_value=(0, 0), bounds_error=False)
        # use interpolation function returned by `interp1d`
        frequency_start = frequency_grid[-1]
        frequency_end = frequency_grid[0]
        frequency_grid_new = np.linspace(frequency_start, frequency_end, self.number_of_grid_points) # frequency grid [Hz]
        frequency_amplitude_new = f(frequency_grid_new)
        # Calculate the amplitude constant A
        A = np.sqrt(self.energy_per_pulse)  
        # Normalize the amplitude profile
        frequency_amplitude_new /= np.sum(np.abs(frequency_amplitude_new))
        # Calculate the frequency step size
        frequency_step = np.abs(frequency_grid_new[1]-frequency_grid_new[0]) * 2*np.pi
        # Scale by the square root of normalized values
        frequency_amplitude_new = np.sqrt(frequency_amplitude_new/self.frequency_bandwidth)
        # Multiply by the amplitude constant to scale to the desired amplitude
        frequency_amplitude_new *= A
        return frequency_grid_new, frequency_amplitude_new
    
    def frequency_to_wavelength_grid(self, 
                                     frequency_grid: np.ndarray[float], 
                                     frequency_amplitude: np.ndarray[float], 
                                     ) -> tuple[np.ndarray[float], np.ndarray[complex]]:
        """
        Converts a grid of frequencies and their corresponding amplitudes to a wavelength grid and amplitudes,
        taking into account the medium's refractive index.

        Parameters:
            frequency_grid (np.ndarray[float]): Array of frequency values.
            frequency_amplitude (np.ndarray[float]): Array of corresponding amplitude values.

        Returns:
            tuple[np.ndarray[float], np.ndarray[complex]]: The new wavelength grid and corresponding amplitude values.
        """
        wavelength_grid = self.convert_frequency_to_wavelength(frequency_grid)
        wavelength_amplitude = frequency_amplitude
        f = interp1d(wavelength_grid, wavelength_amplitude, fill_value=(0, 0), bounds_error=False)
        # use interpolation function returned by `interp1d`
        wavelength_start = wavelength_grid[-1]
        wavelength_end = wavelength_grid[0]
        wavelength_grid_new = np.linspace(wavelength_start, wavelength_end, self.number_of_grid_points) # frequency grid [Hz]
        wavelength_amplitude_new = f(wavelength_grid_new)
        # Calculate the amplitude constant A
        A = np.sqrt(self.energy_per_pulse)  
        # Normalize the amplitude profile
        wavelength_amplitude_new /= np.sum(np.abs(wavelength_amplitude_new))
        # Calculate the wavelngth step size
        wavelength_step = np.abs(self.wavelength_grid[1]-self.wavelength_grid[0])
        # Scale by the square root of normalized values
        wavelength_amplitude_new = np.sqrt(wavelength_amplitude_new/self.wavelength_bandwidth)
        # Multiply by the amplitude constant to scale to the desired amplitude
        wavelength_amplitude_new *= A
        return wavelength_grid_new, wavelength_amplitude_new
    
    def get_time_amplitude(self) -> tuple[np.ndarray[float], np.ndarray[complex]]:
        """
        Converts the pulse's spectral amplitude into the time domain using an inverse Fourier transform,
        to obtain the electric field as a function of time.

        Returns:
            tuple[np.ndarray[float], np.ndarray[complex]]: Time grid and corresponding electric field amplitudes.
        """
        # Perform an inverse Fourier transform to get the time-domain signal
        time_amplitude = np.fft.ifft(self.frequency_amplitude)

        # The frequencies array helps to set the correct time scale
        # Calculate time step (assuming frequencies are evenly spaced)
        frequency_spacing = self.frequency_grid[1] - self.frequency_grid[0]
        # Generate time vector
        time_grid = np.fft.fftfreq(len(self.frequency_grid), d=frequency_spacing)

        return time_grid, time_amplitude

    def calculate_phase_shift(self,
                               optical_path_difference: float,
                               ) -> float:
        """
        Calculates the phase shift experienced by the pulse due to an optical path difference
        in a medium with a given refractive index.

        Parameters:
            optical_path_difference (float): The optical path difference in meters.

        Returns:
            float: The phase shift in radians.
        """
        return (2 * np.pi / self.wavelength_central) * self.refractive_index_function(2*np.pi*c/self.wavelength_central) * optical_path_difference

    def distort_pulse_spectrum(self, bandwidth_omega: float, distortion_function: Callable[[np.ndarray[complex]], np.ndarray[complex]]) -> np.ndarray[complex]:
        """
        Applies a spectral distortion to the pulse using a provided distortion function,
        which can modify the pulse's amplitude and/or phase within a specified bandwidth.

        Parameters:
            distortion_function (Callable[[np.ndarray[complex]], np.ndarray[complex]]): 
                Function that defines the spectral distortion to be applied.
            bandwidth_omega float: bandwidth over wich distortion is applied

        Returns:
            np.ndarray[complex]: The distorted spectral amplitude of the pulse.
        """
        central_omega = 2*np.pi * self.frequency_central
        omega = 2 * np.pi * self.frequency_grid

        # Apply distortion within the specified bandwidth
        return self.frequency_amplitude * distortion_function(omega, central_omega, bandwidth_omega)

    def random_distortion(self, omega: np.ndarray[float], central_omega: float, bandwidth_omega: float, phase_variance: float=1, amplitude_variance: float=0.1) -> np.ndarray[complex]:
        """
        Applies a random spectral distortion to the pulse within a specified bandwidth,
        characterized by random variations in amplitude and phase.

        Parameters:
            omega (np.ndarray[float]): Angular frequency array.
            central_omega (float): Central angular frequency of the pulse.
            bandwidth_omega (float): Bandwidth in angular frequency within which the distortion is applied.
            phase_variance (float): Variance of the random phase shifts applied within the bandwidth.
            amplitude_variance (float): Variance of the random amplitude changes applied within the bandwidth.

        Returns:
            np.ndarray[complex]: The factor representing the random spectral distortion.
        """
        np.random.seed(0)  # For reproducibility, you can remove or change the seed value for different random results

        # Initialize distortion factor with no distortion
        distortion_factor = np.ones_like(omega, dtype=complex)

        # Define the frequency range for applying distortion
        lower_bound = central_omega - bandwidth_omega / 2
        upper_bound = central_omega + bandwidth_omega / 2

        # Apply distortion only within the specified bandwidth
        within_bandwidth = (omega >= lower_bound) & (omega <= upper_bound)
        distortion_factor[within_bandwidth] *= (
            (1 + np.random.normal(0, amplitude_variance, np.sum(within_bandwidth))) *
            np.exp(1j * np.random.normal(0, phase_variance, np.sum(within_bandwidth)))
        )

        return distortion_factor

    def add_quantum_noise(self) -> np.ndarray[complex]:
        """
        Adds quantum noise to the pulse, simulating the effect of quantum fluctuations
        on the pulse's intensity and phase.

        Returns:
            np.ndarray[complex]: The electric field of the pulse with added quantum noise.
        """
        
        # This is all to get the number of photons/second in each frequency bin:
        size_of_bins = self.frequency_grid[1] - self.frequency_grid[0] # Bin width in [Hz]
        power_per_bin = np.abs(self.frequency_amplitude)**2 / size_of_bins # [J]
                    
        photon_energy = h * self.frequency_grid # h nu [J]
        photons_per_bin = power_per_bin/photon_energy # photons / second
        photons_per_bin[photons_per_bin<0] = 0 # must be positive.
        
        # now generate some random intensity and phase arrays:
        size = np.shape(self.frequency_amplitude)[0]
        random_intensity = np.random.normal(size=size)
        random_phase = np.random.uniform(size=size) * 2 * np.pi
        
        noise = random_intensity * np.sqrt(photons_per_bin) * photon_energy * size_of_bins * np.exp(1j*random_phase)
        
        return noise


    def apply_gdd_to_pulse(self, gdd_fs2: float) -> np.ndarray[complex]:
        """
        Applies Group Delay Dispersion (GDD) to the pulse, modifying its phase in the frequency domain
        based on the specified GDD parameter.

        Parameters:
            gdd_fs2 (float): Group Delay Dispersion parameter in femtoseconds squared.

        Returns:
            np.ndarray[complex]: The electric field of the pulse after applying GDD.
        """
        omega_0 = 2 * np.pi * self.frequency_central  # Central angular frequency
        omega = 2 * np.pi * self.frequency_grid  # Angular frequency array

        # Apply GDD in frequency domain
        phase_shift = 0.5 * gdd_fs2 * 1e-30 * (omega - omega_0)**2  # Convert fs^2 to s^2
        return self.frequency_amplitude * np.exp(1j * phase_shift)