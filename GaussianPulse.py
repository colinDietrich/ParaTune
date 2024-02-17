import Pulse
import jax.numpy as np


class GaussianPulse(Pulse):
    def __init__(self, n, temp_span, 
                 power, wl_central, temp_BW, 
                 frep = 60*1e6, power_is_avg = False):
        
        Pulse.__init__(self, n, temp_span=temp_span, wl_central=wl_central, temp_BW=temp_BW)
        self.set_temp_grid()
        self.set_temp_step()
        
        # from https://www.rp-photonics.com/gaussian_pulses.html
        self.temp_amp = np.sqrt(power) * np.exp(-2.77*0.5*self.temp_grid**2/(self.temp_BW**2)) # input field (W^0.5)
        if power_is_avg:            
            self.temp_amp = self.temp_amp * np.sqrt( power / ( frep * self.calulate_energy_per_pulse()) )
        
        self.update_freq_parameters()
        self.update_wl_parameters()

        def transform_limited_pulse_BW(self, pulse_duration):
            return 0.44/pulse_duration