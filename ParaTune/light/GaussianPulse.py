import numpy as np
from ParaTune.light.Pulse import Pulse
from typing import Callable
from scipy.constants import c, pi

class GaussianPulse(Pulse):
    """
    Represents a Gaussian optical pulse, characterized by a Gaussian-shaped amplitude
    in both the frequency and time domains.

    Attributes:
        wavelength_central (float): Central wavelength of the Gaussian pulse in meters.
        wavelength_bandwidth (float): Spectral bandwidth of the Gaussian pulse in meters.
        mean_power (float): Average power of the Gaussian pulse in Watts.
        repetition_rate (float): Repetition rate of the Gaussian pulse in Hertz.
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
                 ) -> None:
        super().__init__(wavelength_central, 
                         wavelength_bandwidth, 
                         mean_power, 
                         repetition_rate, 
                         number_of_grid_points, 
                         wavelength_span, 
                         refractive_index_function)

    @property
    def wavelength_amplitude(self) -> np.ndarray:
        """
        Calculates the amplitude of the Gaussian pulse as a function of wavelength.

        Returns:
            np.ndarray: Array of complex numbers representing the amplitude of the Gaussian pulse across the wavelength grid.
        """
        # Define constants for Gaussian formula
        sigma = self.wavelength_bandwidth / 2.35482

        # Calculate the amplitude constant A
        A = np.sqrt(self.energy_per_pulse)      

        # Calculate the Gaussian profile for amplitude
        wavelength_amplitude = np.exp(-((self.wavelength_grid - self.wavelength_central) ** 2) / (2 * sigma**2))

        # Normalize the Gaussian profile
        wavelength_amplitude /= np.sum(np.abs(wavelength_amplitude))

        # Calculate the wavelngth step size
        wavelength_step = np.abs(self.wavelength_grid[1]-self.wavelength_grid[0])

        # Scale by the square root of normalized values
        wavelength_amplitude = np.sqrt(wavelength_amplitude/self.wavelength_bandwidth)

        # Multiply by the amplitude constant to scale to the desired amplitude
        wavelength_amplitude *= A
        return wavelength_amplitude

    @property
    def time_bandwidth(self) -> float:
        """
        Calculates the time-bandwidth product of the Gaussian pulse.

        Returns:
            float: The time-bandwidth product of the Gaussian pulse.
        """
        return 0.44 / (2 * np.pi * self.frequency_bandwidth)
