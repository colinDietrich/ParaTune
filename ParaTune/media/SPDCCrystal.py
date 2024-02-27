import numpy as np
from typing import List, Optional
from scipy.constants import c
from ParaTune.media.Crystal import Crystal
from ParaTune.media.data import *

class SPDCCrystal(Crystal):
    def __init__(self, 
                 configuration: str, 
                 medium: str, 
                 number_grid_points_z: int, 
                 wavelength_central: float, 
                 signal: str, 
                 idler: str, 
                 pump: str, 
                 domain_width: Optional[float] = None, 
                 length: Optional[float] = None,
                 maximum_length: Optional[float] = None, 
                 minimum_length: Optional[float] = None,
                 domain_values_custom: Optional[List[int]] = None,
                 domain_bounds_custom: Optional[List[float]] = None
                 ) -> None:
        
        self.signal = signal # polarization axis of signal field
        self.idler = idler # polarization axis of idler field
        self.pump = pump # polarization axis of pump field
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
        
        # Initialization crystal's domain width (if it is not set yet)
        if(domain_width is None): 
            dk = self.wavevector_mismatch()(self.angular_frequency_central/2, self.angular_frequency_central/2)
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
        return lambda ws, wi: self.k_s(ws)+self.k_i(wi)-self.k_p(ws+wi)