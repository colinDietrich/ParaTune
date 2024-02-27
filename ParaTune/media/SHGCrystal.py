import numpy as np
from typing import List, Optional
from scipy.constants import c
from ParaTune.media.Crystal import Crystal
from ParaTune.media.data import *

class SHGCrystal(Crystal):
    def __init__(self, 
                 configuration: str, 
                 medium: str, 
                 number_grid_points_z: int, 
                 wavelength_central: float, 
                 fundamental: str, 
                 second_harmonic: str, 
                 domain_width: Optional[float] = None, 
                 length: Optional[float] = None,
                 maximum_length: Optional[float] = None, 
                 minimum_length: Optional[float] = None,
                 domain_values_custom: Optional[List[int]] = None,
                 domain_bounds_custom: Optional[List[float]] = None
                 ) -> None:
        self.fundamental = fundamental
        self.second_harmonic = second_harmonic
        self.wavelength_central = wavelength_central # Pump central wavelength [m]
        self.angular_frequency_central = 2*np.pi*c/wavelength_central # Pump central angular frequency [rad Hz]

        # initialization of crystal medium
        if(medium == 'LiNbO3'):
            self.data_medium = LiNbO3
            self.orientations = ['x', 'y']
        elif(medium == 'KTP'):
            self.data_medium = KTP
            self.orientations = ['x', 'y', 'z']
        else:
            raise ValueError('Unkown medium for the crystal.')
        
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

        # Initialization crystal's domain width (if it is not set yet)
        if(domain_width is None): 
            dk = self.wavevector_mismatch()(self.angular_frequency_central, self.angular_frequency_central, self.angular_frequency_central*2)
            domain_width = np.abs(np.pi / dk) 

        super().__init__(configuration, 
                        medium, 
                        number_grid_points_z, 
                        domain_width, 
                        length,
                        maximum_length, 
                        minimum_length,
                        domain_values_custom,
                        domain_bounds_custom)

    def wavevector_mismatch(self):
        return lambda wf1, wf2, wsh:  self.k_f(wf1) + self.k_f(wf2) - self.k_sh(wsh)