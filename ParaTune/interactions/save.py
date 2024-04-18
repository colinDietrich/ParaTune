import jax.numpy as np
import jax
from functools import partial
from scipy.constants import epsilon_0, c
from jax.scipy.signal import correlate as corr
from jax.scipy.signal import fftconvolve as conv

class SHG:
    def __init__(self, wl_central, freq_span, length,
                 n_w, n_z, n_f, n_sh, k_f, k_sh, wavevector_mismatch,
                 d_eff, domain_bounds, tau):

        self.lambda_f_0 = wl_central  # wavelength of pump/fundamental [m]
        self.omega_f_0 = (2*np.pi*c)/self.lambda_f_0 # signal mode central frequency [rad/s]
        self.deff = d_eff  # d_eff [m/V]
        self.domain_bounds = domain_bounds # positions of the boundaries of each domain
        self.tau = tau # FWHM pulse duration [s]
        self.sell_f = n_f # Sellmeier equation for fundamental field polarization
        self.sell_sh = n_sh # Sellmeier equation for second harmonic field polarization
        self.k_f = k_f  # wave-vectors as a function of frequency along polarization of fundamental
        self.k_sh = k_sh    # wave-vectors as a function of frequency along polarization of second harmonic
        self.dk = wavevector_mismatch   # wavevector mismatch function
        self.W = freq_span # spectral window
        self.Z = length # Z dimension of the crystal
        self.Nw = n_w # number of samples along frequency
        self.Nz = n_z # number of samples along z

        self.init_grid()


    def init_grid(self):
        # initialize the grid
        self.dz = self.Z / self.Nz
        self.z = np.linspace(0, self.Z, self.Nz)

        self.w_f = np.linspace(self.omega_f_0 - self.W/2,self.omega_f_0 + self.W/2, self.Nw)
        self.w_sh = np.linspace(self.omega_f_0*2 - self.W/2,self.omega_f_0*2 + self.W/2, self.Nw)
        self.dw = self.w_f[1] - self.w_f[0]

        self.D_f = np.gradient(self.k_f(self.w_f), self.dw)
        self.GVD_f = np.gradient(self.k_f(self.D_f), self.dw)

        self.D_sh = np.gradient(self.k_sh(self.w_sh), self.dw)
        self.GVD_sh = np.gradient(self.k_sh(self.D_sh), self.dw)

        self.WW_sh, self.ZZ_sh = np.meshgrid(self.w_sh, self.z)
        self.WW_f, self.ZZ_f = np.meshgrid(self.w_f, self.z)

    def SHG_NCME(self, _z, _A, _d):
        # nonlinear coupled mode equations of SHG / ODE that describes the nonlinear interaction
        A1 = _A[0]
        A2 = _A[1]

        a1 = 1j * self.w_f / self.sell_f(self.w_f) / c * self.deff * _d * corr(A1 * np.exp(1j * _z *self.k_f(self.w_f)), A2 * np.exp(1j * _z *self.k_sh(self.w_sh)), mode='same', method='fft') * np.exp(-1j * _z *self.k_f(self.w_f))
        a2 = 1j * self.w_sh / 2 / self.sell_sh(self.w_sh) / c * self.deff * _d * conv(A1 * np.exp(1j * _z *self.k_f(self.w_f)), A1 * np.exp(1j * _z *self.k_f(self.w_f)), mode='same') * np.exp(-1j * _z *self.k_sh(self.w_sh))

        return np.array([a1, a2])

    def nonlinear_step(self, _z0, _d, _A1, _A2):
        # do a single nonlinear step by solving the SHG ODE with runge-kutta
        sol = self.runge_kutta_method(
            _z0,
            self.SHG_NCME,
            np.asarray([_A1, _A2]),
            self.dz,
            _d,
        )
        return sol[0], sol[1]

    def runge_kutta_method(self, _z0, f, A0, h, _d):
        # 4th order Runge-Kutta method for step length of h
        F1 = h * f(_z0, A0, _d)
        F2 = h * f(_z0 + (h / 2), (A0 + F1 / 2), _d)
        F3 = h * f(_z0 + (h / 2), (A0 + F2 / 2), _d)
        F4 = h * f(_z0 + h, (A0 + F3), _d)

        y1 = A0 + 1 / 6 * (F1 + 2 * F2 + 2 * F3 + F4)
        return y1

    def save_energy(self, _Af, _Ash):
        # save energy of fundamental and second harmonic
        E_f = np.asarray(
            np.sum(np.abs(_Af) ** 2)
            * self.tau
            * (self.sell_f(self.omega_f_0) * c * epsilon_0)
            / 2
        )
        E_sh = np.asarray(
            np.sum(np.abs(_Ash) ** 2)
            * self.tau
            * (self.sell_sh(self.omega_f_0/2) * c * epsilon_0)
            / 2
        )

        return E_f, E_sh

    @partial(jax.jit, static_argnums=(0,))
    def run(self, _A1, _p):
        # initial fields
        A1 = np.array(_A1, dtype=np.complex64)  # Spectral amplitude of fundamental
        A2 = np.array(np.zeros_like(A1), dtype=np.complex64)  # Spectral amplitude of second harmonic
        E1 = np.array(np.zeros(self.Nz), dtype=np.float32)  # energy of fundamental along z
        E2 = np.array(np.zeros(self.Nz), dtype=np.float32)  # energy of second harmonic along z
        d = _p

        # loop over all steps along z
        @partial(jax.jit, static_argnums=0)
        def body_fun(n, val):
            a1, a2 = self.nonlinear_step(self.z[n], d[n], val[0], val[1])
            val_e1, val_e2 = self.save_energy(a1, a2)
            e1 = val[2][:]
            e2 = val[3][:]
            e1 = e1.at[n].set(val_e1)
            e2 = e2.at[n].set(val_e2)
            res = [a1,a2,e1,e2]
            return res
        A = jax.lax.fori_loop(0, self.Nz, body_fun, [A1,A2,E1,E2])

        return A