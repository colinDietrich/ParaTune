from scipy.constants import c
import math
from scipy.special import genlaguerre, hermite
from typing import List, Optional, Callable
import numpy as np

class Beam:
    """
    Represents a beam of light in 3d.

    Attributes:
        width (float): The width of the beam.
        num_points_x (int): The number of points in the x direction.
        height (float): The height of the beam.
        num_points_y (int): The number of points in the y direction.
        angular_frequency_grid (List[float]): The grid of angular frequencies.
        angular_frequency_amplitude (List[float]): The amplitude of the angular frequencies.
        wave_number_func (Callable[[List[float]], List[float]]): The function to calculate wave numbers.
        refractive_index_func (Callable[[List[float]], List[float]]): The function to calculate refractive indices.
        beam_waist (float): The beam waist.
        mode (str): The mode of the beam. Default is 'Lg'.
        initial_position (float): The initial position of the beam. Default is 0.
        max_azimuthal_index (Optional[int]): The maximum azimuthal index. Default is None.
        max_radial_index (Optional[int]): The maximum radial index. Default is None.
        core_refractive_index (Optional[float]): The refractive index of the core. Default is None.
        cladding_refractive_index (Optional[float]): The refractive index of the cladding. Default is None.
        core_radius (Optional[float]): The radius of the core. Default is None.
    """

    def __init__(self, 
                 width: float, 
                 num_points_x: int, 
                 height: float, 
                 num_points_y: int,
                 angular_frequency_grid: List[float], 
                 angular_frequency_amplitude: List[float],
                 wave_number_func: Callable[[List[float]], List[float]],
                 refractive_index_func: Callable[[List[float]], List[float]], 
                 beam_waist: float,
                 mode: str = 'Lg', 
                 initial_position: float = 0, 
                 max_azimuthal_index: Optional[int] = None,
                 max_radial_index: Optional[int] = None, 
                 core_refractive_index: Optional[float] = None,
                 cladding_refractive_index: Optional[float] = None, 
                 core_radius: Optional[float] = None,
                 dimensions: int = 3):
        
        self.width = width
        self.num_points_x = num_points_x
        self.height = height
        self.num_points_y = num_points_y
        self.angular_frequency_grid = angular_frequency_grid
        self.angular_frequency_amplitude = angular_frequency_amplitude
        self.wave_number_func = wave_number_func
        self.refractive_index_func = refractive_index_func
        self.beam_waist = beam_waist
        self.mode = mode
        self.initial_position = initial_position
        self.max_azimuthal_index = max_azimuthal_index
        self.max_radial_index = max_radial_index

        self.x = np.linspace(-self.width/2, self.width/2, self.num_points_x)
        self.y = np.linspace(-self.height/2, self.height/2, self.num_points_y)
        self.XX, self.YY, self.WW = np.meshgrid(self.x, self.y, self.angular_frequency_grid, indexing='ij')

        ones_3d = np.ones((self.num_points_x, self.num_points_y, len(self.angular_frequency_grid)))
        self.KK = np.multiply(ones_3d, self.wave_number_func(self.angular_frequency_grid))
        self.NN = np.multiply(ones_3d, self.refractive_index_func(self.angular_frequency_grid))

        if max_azimuthal_index is None or max_radial_index is None:
            if all(v is not None for v in [core_refractive_index, cladding_refractive_index, core_radius]):
                self.max_azimuthal_index = self.max_radial_index = self.get_max_modes_number(core_refractive_index, cladding_refractive_index, core_radius, self.angular_frequency_grid)
                print(f'Max number of modes propagating in medium = {self.max_azimuthal_index}')
            else:
                raise ValueError('Insufficient parameters to determine the maximum number of modes.')

        if self.mode == 'Lg':
            self.Beam_array = [self.laguerre_gaussian_mode(n, m) for n in range(self.max_azimuthal_index) for m in range(self.max_radial_index)]
        elif self.mode == 'HG':
            self.Beam_array = [self.hermite_gaussian_mode(n, m) for n in range(self.max_azimuthal_index) for m in range(self.max_radial_index)]
        else:
            raise ValueError('Unknown mode name.')
        
        # Set up the dimensions of the simulation
        self.dimension = dimensions
        # Normalize the Beam array according to the dimension
        for i in range(len(self.Beam_array)):
            if(self.dimension == 2):
                # Calculate the x step size
                x_step = np.abs(self.x[1] - self.x[0])
                self.Beam_array[i] = np.sum(self.Beam_array[i], axis=0)
                # Normalize the Spatial profile to have unit area
                self.Beam_array[i] = self.Beam_array[i] / np.sum(np.abs(self.Beam_array[i]), axis=0)
                # Scale by the square root of normalized values
                self.Beam_array[i] = np.sqrt(self.Beam_array[i])
                # Multiply by frequency amplitudes calulated previoulsy
                self.Beam_array[i] *= self.angular_frequency_amplitude[np.newaxis, :]
            else :
                # Calculate the x / y step size
                x_step = np.abs(self.x[1] - self.x[0])
                y_step = np.abs(self.y[1] - self.y[0])
                # Normalize the Spatial profile to have unit area
                self.Beam_array[i] = self.Beam_array[i] / np.sum(np.abs(self.Beam_array[i]), axis=(0, 1))
                # Scale by the square root of normalized values
                self.Beam_array[i] = np.sqrt(self.Beam_array[i]/self.beam_waist**2)
                # Multiply by frequency amplitudes calulated previoulsy
                self.Beam_array[i] *= self.angular_frequency_amplitude[np.newaxis, np.newaxis, :]
        
    def laguerre_gaussian_mode(self, azimuthal_index: int, radial_index: int, coef: Optional[float] = None) -> np.ndarray:
            """
            Calculates the Laguerre-Gaussian mode for the given azimuthal and radial indices.

            Parameters:
            - azimuthal_index (int): The azimuthal index of the mode.
            - radial_index (int): The radial index of the mode.
            - coef (Optional[float]): The coefficient of the mode. If not provided, it is calculated based on the indices.

            Returns:
            - np.ndarray: The Laguerre-Gaussian mode as a numpy array.
            """

            rayleigh_range = self.KK / 2 * self.beam_waist**2 * self.NN
            beam_radius = self.beam_waist * np.sqrt(1 + (self.initial_position / rayleigh_range)**2)

            r = np.sqrt(self.XX**2 + self.YY**2)
            phi = np.arctan2(self.XX, self.YY)

            invR = self.initial_position / (self.initial_position**2 + rayleigh_range**2)
            gouy = (np.abs(radial_index) + 2 * azimuthal_index + 1) * np.arctan(self.initial_position / rayleigh_range)

            if coef is None:
                coef = np.sqrt(2 * math.factorial(azimuthal_index) / (np.pi * math.factorial(azimuthal_index + np.abs(radial_index))))

            laguerre = (coef * (self.beam_waist / beam_radius) * (r * np.sqrt(2) / beam_radius)**np.abs(radial_index) *
                       np.exp(-r**2 / beam_radius**2) * np.array(genlaguerre(azimuthal_index, abs(radial_index))(2 * r**2 / beam_radius**2)) *
                       np.exp(-1j * (self.KK * r**2 / 2) * invR) * np.exp(-1j * radial_index * phi) * np.exp(1j * gouy))

            return laguerre


    def hermite_gaussian_mode(self, n: int, m: int) -> np.ndarray:
        """
        Calculate the Hermite-Gaussian mode for given transverse indices.

        Parameters:
            n (int): Order of the Hermite polynomial in the x direction.
            m (int): Order of the Hermite polynomial in the y direction.

        Returns:
            np.ndarray: Computed Hermite-Gaussian mode field distribution.
        """
        hx = hermite(n)(self.x * np.sqrt(2) / self.beam_waist)
        hy = hermite(m)(self.y * np.sqrt(2) / self.beam_waist)
        Hx = np.outer(hx, np.ones_like(self.y)).reshape((self.num_points_x, self.num_points_y, 1))
        Hy = np.outer(np.ones_like(self.x), hy).reshape((self.num_points_x, self.num_points_y, 1))
        
        amplitude = np.exp(-(self.XX**2 + self.YY**2) / self.beam_waist**2)
        phase = np.exp(-1j * (n + m + 1) * np.arctan(self.initial_position / (self.KK / 2 * self.beam_waist**2 * self.NN)))
        
        return Hx * Hy * amplitude * phase
