import numpy as np
from ParaTune.light.Pulse import Pulse
from typing import Callable

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
        refractive_index_function (Callable[[float], float]): Function to calculate the refractive index as a function of wavelength.
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
        wavelength_amplitude = np.zeros_like(self.wavelength_grid, dtype=complex)
        amplitude = np.sqrt(self.average_photons_per_pulse)
        central_omega = 2 * np.pi * self.convert_wavelength_to_frequency(self.wavelength_central)
        bandwidth_omega = 2 * np.pi * self.get_frequency_bandwidth()
        omega = 2 * np.pi * self.convert_wavelength_to_frequency(self.wavelength_grid)
        wavelength_amplitude += amplitude * np.exp(-((omega - central_omega) ** 2) / (2 * bandwidth_omega ** 2))
        return wavelength_amplitude

    @property
    def time_bandwidth(self) -> float:
        """
        Calculates the time-bandwidth product of the Gaussian pulse.

        Returns:
            float: The time-bandwidth product of the Gaussian pulse.
        """
        return 0.44 / (2 * np.pi * self.frequency_bandwidth)
