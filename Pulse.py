import jax.numpy as np
from scipy.constants import h, c
from scipy.interpolate import interp1d


class Pulse:

    def __init__(self, n, wl_central=None, wl_span=None, wl_BW=None,
                freq_central=None, freq_span=None, freq_BW=None,
                temp_span=None, temp_BW=None):
        
        self.n = n # number of grid points
        # wavelength parameters
        self.wl_step = None  # discretization step on wavelength grid [m]
        self.wl_central = None  # central wavelength [m]
        self.wl_span = None # wavelength window [m]
        self.wl_amp = None  # array of amplitudes of the field along wavelength grid [V/m]
        self.wl_grid = None # wavelength grid [m]
        self.wl_BW = None # wavelength bandwidth of the pulse [m]
        # frequency parameters
        self.freq_step = None  # discretization step on frequency grid [Hz]
        self.freq_central = None  # central frequency [Hz]
        self.freq_span = None # frequency window [Hz]
        self.freq_amp = None  # array of amplitudes of the field along frequency grid [V/Hz]
        self.freq_grid = None # frequency grid [Hz]
        self.wl_BW = None # spectral bandwidth of the pulse [Hz]
        # temporal parameters
        self.temp_step = None  # discretization step on frequency grid [Hz]
        self.temp_span = None # frequency window [Hz]
        self.temp_amp = None  # array of amplitudes of the field along frequency grid [V/Hz]
        self.temp_grid = None # frequency grid [Hz]
        self.temp_BW = None # temporal bandwidth of the pulse [s]


        # wavelength parameters of the pulse
        if(wl_span is not None):
            self.wl_span = wl_span
            if(wl_BW is not None): self.wl_BW = wl_BW
        if(wl_central is not None): 
            self.wl_central = wl_central
        # frequency parameters of the pulse
        elif(freq_span is not None):
            self.freq_span = freq_span
            if(freq_BW is not None): self.freq_BW = freq_BW
        if(freq_central is not None):
            self.freq_central = freq_central
        # frequency parameters of the pulse
        elif(temp_span is not None and (wl_central is not None or freq_central is not None)):
            self.temp_span = temp_span
            if(temp_BW is not None): self.temp_BW = temp_BW
            if(wl_central is not None): 
                self.wl_central = wl_central
                self.set_freq_central()
            else: 
                self.freq_central = freq_central
        
    # ----- WAVELENGTH -----

    def set_wl_step(self):
        if(self.wl_grid is not None):
            self.wl_step = self.wl_grid[1]-self.wl_grid[0]
        else:
            raise ValueError('Frequency step is not set.')

    def set_wl_central(self):
        if(self.freq_central is not None):
            self.wl_central = self.freq_to_wl(self.freq_central)
        else:
            raise ValueError('Central frequency is not set.')
        
    def set_wl_grid(self):
        if(self.wl_central is not None and self.wl_span is not None):
            self.wl_grid = np.linspace(self.wl_central - self.wl_span/2, self.wl_central + self.wl_span/2, self.n)
        else:
            raise ValueError('Wavelength grid parameters are not set.')

    def set_wl_span(self):
        if(self.freq_grid is not None):
            self.wl_span = self.freq_grid_span_to_wl(self.freq_grid[-1], self.freq_grid[0])
        else:
            raise ValueError('Frequency span is not set.')

    def set_wl_amp(self):
        if(self.freq_amp is not None and self.freq_grid is not None and self.wl_grid is not None):
            self.wl_amp = self.freq_grid_to_wl_grid(self.freq_grid, self.freq_amp, self.wl_grid)
        else:
            raise ValueError('Frequency amplitude is not set.')
        
    def update_wl_parameters(self):
        self.set_wl_central()
        self.set_wl_span()
        self.set_wl_grid()
        self.set_wl_step()
        self.set_wl_amp()
        
    # ----- FREQUENCY -----

    def set_freq_step(self):
        if(self.freq_grid is not None):
            self.freq_step = self.freq_grid[1]-self.freq_grid[0]
        elif(self.temp_step is not None):
            self.freq_step = 1 / self.temp_step / self.n
        else:
            raise ValueError('Wavelength / Temporal step is not set.')

    def set_freq_central(self):
        if(self.wl_central is not None):
            self.freq_central = self.wl_to_freq(self.wl_central)
        else:
            raise ValueError('Central wavelength is not set.')

    def set_freq_span(self):
        if(self.wl_grid is not None):
            self.freq_span = self.wl_grid_span_to_freq(self.wl_grid[0], self.wl_grid[-1])
        elif(self.temp_span is not None):
            self.freq_span = 1/self.temp_step
        else:
            raise ValueError('Wavelength / Temporal span is not set.')
        
    def set_freq_grid(self):
        if(self.freq_central is not None and self.freq_span is not None):
            self.freq_grid = np.linspace(self.freq_central - self.freq_span/2, self.freq_central + self.freq_span/2, self.n)
        else:
            raise ValueError('Frequency grid parameters are not set.')

    def set_freq_amp(self):
        if(self.wl_grid is not None and self.wl_amp is not None and self.freq_grid is not None):
            self.freq_amp = self.wl_grid_to_freq_grid(self.wl_grid, self.wl_amp, self.freq_grid)
        elif(self.temp_amp is not None):
            self.freq_amp = (np.fft.ifftshift(np.fft.ifft(self.temp_amp)))
        else:
            raise ValueError('Wavelength / Temporal amplitude is not set.')
        
    def update_freq_parameters(self):
        self.set_freq_central()
        self.set_freq_span()
        self.set_freq_grid()
        self.set_freq_step()
        self.set_freq_amp()
        
    # ----- TIME -----

    def set_temp_step(self):
        if(self.temp_grid is not None):
            self.temp_step = self.temp_grid[1]-self.temp_grid[0]
        elif(self.freq_step is not None):
            self.temp_step = 1 / self.freq_step / self.n
        else:
            raise ValueError('Frequency step is not set.')

    def set_temp_span(self):
        if(self.temp_step is not None):
            self.temp_span = (self.n) * self.temp_step
        else:
            raise ValueError('Temporal step is not set.')
        
    def set_temp_grid(self):
        if(self.temp_span is not None):
            self.temp_grid = np.linspace(-self.temp_span/2, self.temp_span/2, self.n)
        else:
            raise ValueError('Temporal grid parameters are not set.')

    def set_temp_amp(self):
        if(self.freq_amp is not None):
            self.temp_amp = np.fft.ifftshift(np.fft.ifft(self.freq_amp))
        else:
            raise ValueError('Frequency amplitude of pulse is not set.')
        
    def update_temp_parameters(self):
        self.set_temp_step()
        self.set_temp_span()
        self.set_temp_grid()
        self.set_temp_amp()

    def wl_grid_span_to_freq(self, wl_1, wl_2):
        return np.abs(self.wl_to_freq(wl_1)-self.wl_to_freq(wl_2))

    def freq_grid_span_to_wl(self, freq_1, freq_2):
        return np.abs(self.freq_to_wl(freq_1)-self.freq_to_wl(freq_2))
    
    def freq_BW_to_wl_BW(self, BW, l0):
        return (4*np.pi*c / BW) * (np.sqrt(1 + ((l0*BW) / (2*np.pi*c))**2) - 1)

    def wl_BW_to_freq_BW(self, BW, l0):
        return 2*np.pi*c * BW * (1/(l0**2 - BW**2 / 4))

    def freq_BW_to_temp_BW(self, BW):
        return 0.5/BW

    def temp_BW_to_freq_BW(self, BW):
        return 0.5/BW

    def wl_to_freq(self, l):
        return (2*np.pi*c)/l
    
    def freq_to_wl(self, w):
        return (2*np.pi*c)/w
    
    def wl_grid_to_freq_grid(self, _wl_grid, _wl_amp, _freq_grid_new):
        _freq_grid = self.wl_to_freq(_wl_grid)
        _freq_amp = _wl_amp[::-1]
        f = interp1d(_freq_grid, _freq_amp, fill_value=(0, 0), bounds_error=False)
        # use interpolation function returned by `interp1d`
        _freq_amp_new = f(_freq_grid_new)
        return _freq_amp_new
    
    def freq_grid_to_wl_grid(self, _freq_grid, _freq_amp, _wl_grid_new):
        _wl_grid = self.freq_to_wl(_freq_grid)
        _wl_amp = _freq_amp[::-1]
        f = interp1d(_wl_grid, _wl_amp, fill_value=(0, 0), bounds_error=False)
        # use interpolation function returned by `interp1d`
        _wl_amp_new = f(_wl_grid_new)
        return _wl_amp_new
    

    def calulate_energy_per_pulse(self):
        """ 
        Calculate and return energy per pulse via numerical integration
        of :math:`A^2 dt`
        
        Returns
        -------
        x : float
            Pulse energy [J]
        """
        return self.temp_step * np.trapz(abs(self.temp_amp)**2)
    
    def set_energy_per_pulse(self, energy):
        """ 
        Set the energy per pulse (in Joules)
            
        Parameters
        ----------
        desired_epp_J : float
                the value to set the pulse energy [J]
                
        Returns
        -------
        nothing
        """
        self.temp_amp = self.temp_amp * np.sqrt(energy / self.calulate_energy_per_pulse())
    

    def add_noise(self, noise_type='sqrt_N_freq'):
        """ 
        Adds random intensity and phase noise to a pulse. 
        
        Parameters
        ----------
        noise_type : string
             The method used to add noise. The options are: 
    
             'sqrt_N_freq' : which adds noise to each bin in the frequency domain, 
             where the sigma is proportional to sqrt(N), and where N
             is the number of photons in each frequency bin. 
    
             'one_photon_freq' : which adds one photon of noise to each frequency bin, regardless of
             the previous value of the electric field in that bin. 
             
        Returns
        -------
        nothing
        """
        
        # This is all to get the number of photons/second in each frequency bin:
        size_of_bins = self.freq_step                          # Bin width in [Hz]
        power_per_bin = np.abs(self.freq_amp)**2 / size_of_bins   # [J*Hz] / [Hz] = [J]
                    
        #photon_energy = h * self.W_THz/(2*np.pi) * 1e12
        photon_energy = h * self.freq_grid # h nu [J]
        photons_per_bin = power_per_bin/photon_energy # photons / second
        photons_per_bin[photons_per_bin<0] = 0 # must be positive.
        
        # now generate some random intensity and phase arrays:
        size = np.shape(self.freq_amp)[0]
        random_intensity = np.random.normal(size=size)
        random_phase = np.random.uniform(size=size) * 2 * np.pi
        
        if noise_type == 'sqrt_N_freq': # this adds Gausian noise with a sigma=sqrt(photons_per_bin)
                                                                      # [J]         # [Hz]
            noise = random_intensity * np.sqrt(photons_per_bin) * photon_energy * size_of_bins * np.exp(1j*random_phase)
        
        elif noise_type == 'one_photon_freq': # this one photon per bin in the frequecy domain
            noise = random_intensity * photon_energy * size_of_bins * np.exp(1j*random_phase)
        else:
            raise ValueError('noise_type not recognized.')
        
        self.freq_amp(self.freq_amp + noise)
        
    
    def chirp_pulse_W(self, GDD, TOD=0, FOD = 0.0, w0_THz = None):
        """ 
        Alter the phase of the pulse 
        
        Apply the dispersion coefficients :math:`\beta_2, \beta_3, \beta_4`
        expanded around frequency :math:`\omega_0`.
        
        Parameters
        ----------
        GDD : float
             Group delay dispersion (:math:`\beta_2`) [ps^2]
        TOD : float, optional
             Group delay dispersion (:math:`\beta_3`) [ps^3], defaults to 0.
        FOD : float, optional
             Group delay dispersion (:math:`\beta_4`) [ps^4], defaults to 0.             
        w0_THz : float, optional
             Center frequency of dispersion expansion, defaults to grid center frequency.
        
        Notes
        -----
        The convention used for dispersion is
        
        .. math:: E_{new} (\omega) = \exp\left(i \left(
                                        \frac{1}{2} GDD\, \omega^2 +
                                        \frac{1}{6}\, TOD \omega^3 +
                                        \frac{1}{24} FOD\, \omega^4 \right)\right)
                                        E(\omega)
                                            
        """                

        if w0_THz is None:
            self.set_AW( np.exp(1j * (GDD / 2.0) * self.V_THz**2 + 
                                   1j * (TOD / 6.0) * self.V_THz**3+ 
                                   1j * (FOD / 24.0) * self.V_THz**4) * self.AW )
        else:
            V = self.W_THz - w0_THz
            self.set_AW( np.exp(1j * (GDD / 2.0) * V**2 + 
                                   1j * (TOD / 6.0) * V**3+ 
                                   1j * (FOD / 24.0) * V**4) * self.AW )
        
                                 
    def dechirp_pulse(self, GDD_TOD_ratio = 0.0, intensity_threshold = 0.05):

        spect_w = self.AW
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl**2 > intensity_threshold * np.max(ampl)**2
        gdd     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 2))
        self.set_AW( ampl * np.exp(1j*(phase-gdd(self.W_THz))) )