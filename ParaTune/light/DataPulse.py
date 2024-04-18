import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Callable
from ParaTune.light.Pulse import Pulse  # Assuming Pulse is correctly imported from ParaTune.light

class DataPulse(Pulse):
    """
    Initializes a DataPulse object with properties of the pulse and the name of the file containing experimental data.

    Parameters:
        wavelength_central (float): Central wavelength of the pulse in meters.
        wavelength_bandwidth (float): Spectral bandwidth of the pulse in meters.
        mean_power (float): Average power of the pulse in Watts.
        repetition_rate (float): Repetition rate of the pulse in Hertz.
        number_of_grid_points (int): Number of grid points for numerical simulations.
        wavelength_span (float): Total span of wavelengths to consider in meters.
        refractive_index_function (Callable[[float], float]): Function to calculate the refractive index as a function of angular frequency.
        file_name (str): Name of the CSV file without the extension, containing the pulse data.
    """

    def __init__(self, 
                 wavelength_central: float, 
                 wavelength_bandwidth: float, 
                 mean_power: float, 
                 repetition_rate: float, 
                 number_of_grid_points: int, 
                 wavelength_span: float, 
                 refractive_index_function: Callable[[float], float],
                 file_name: str
                 ) -> None:
        
        self.file_name = file_name
    
        super().__init__(wavelength_central, 
                         wavelength_bandwidth, 
                         mean_power, 
                         repetition_rate, 
                         number_of_grid_points, 
                         wavelength_span, 
                         refractive_index_function)

    @property
    def wavelength_amplitude(self) -> None:
        """
        Reads the experimental data from the CSV file and interpolates the amplitude values over the wavelength grid.

        Raises:
            ValueError: If the CSV file cannot be found or read properly.
        """
        try:
            df = pd.read_csv(self.file_name + ".csv", on_bad_lines='skip', delimiter=";", decimal=",")
        except FileNotFoundError:
            raise ValueError(f"ERROR: File '{self.file_name}.csv' not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while reading '{self.file_name}.csv': {e}")

        # Data from CSV file
        wavelength_grid = np.array(df.iloc[:, 0])*1e-9
        amplitude_wavelength = np.array(df.iloc[:, 1])

        # Interpolate amplitude values to match the pulse's wavelength grid
        f = interp1d(wavelength_grid, amplitude_wavelength, fill_value=(0, 0), bounds_error=False)

        return f(self.wavelength_grid)
