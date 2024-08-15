import numpy as np
from scipy.signal import correlate, convolve
from scipy.integrate import solve_ivp
from tqdm import tqdm  # For progress bar
import random
from scipy.constants import c, hbar, epsilon_0

def nonlinear_step(z, Ap, As_out, Ai_out, As_vac, Ai_vac, nonlinear_orientation, KK_p, KK_s, KK_i, common_factor_p, common_factor_s, common_factor_i, dimensions):
    # Exponential factors for phase matching
    exp_p, exp_s, exp_i = np.exp(1j * z * KK_p), np.exp(1j * z  * KK_s), np.exp(1j * z * KK_i)
    common_factor_p_b = common_factor_p * nonlinear_orientation
    common_factor_s_b = common_factor_s * nonlinear_orientation
    common_factor_i_b = common_factor_i * nonlinear_orientation

    # Compute the nonlinear interaction based on the number of dimensions
    if dimensions == 1:
        # Apply symmetry in the interaction terms
        As_out = As_out + common_factor_s_b * correlate(Ap * exp_p, Ai_vac * exp_i, mode='same') * np.exp(-1j * z * KK_s)
        As_vac = As_vac + common_factor_s_b * correlate(Ap * exp_p, Ai_out * exp_i, mode='same') * np.exp(-1j * z * KK_s)
        Ai_out = Ai_out + common_factor_i_b * correlate(Ap * exp_p, As_vac * exp_s, mode='same') * np.exp(-1j * z * KK_i)
        Ai_vac = Ai_vac + common_factor_i_b * correlate(Ap * exp_p, As_out * exp_s, mode='same') * np.exp(-1j * z * KK_i)
        Ap = Ap + common_factor_p_b * convolve(As_out * exp_s, Ai_out * exp_i, mode='same') * np.exp(-1j * z * KK_p)
    elif dimensions == 2:
        for i in range(Ap.shape[0]):
            As_out[i, :] = As_out[i, :] + common_factor_s_b[i, :] * correlate(Ap[i, :] * exp_p[i, :], Ai_vac[i, :] * exp_i[i, :], mode='same') * np.exp(-1j * z * KK_s[i, :])
            As_vac[i, :] = As_vac[i, :] + common_factor_s_b[i, :] * correlate(Ap[i, :] * exp_p[i, :], Ai_out[i, :] * exp_i[i, :], mode='same') * np.exp(-1j * z * KK_s[i, :])
            Ai_out[i, :] = Ai_out[i, :] + common_factor_i_b[i, :] * correlate(Ap[i, :] * exp_p[i, :], As_vac[i, :] * exp_s[i, :], mode='same') * np.exp(-1j * z * KK_i[i, :])
            Ai_vac[i, :] = Ai_vac[i, :] + common_factor_i_b[i, :] * correlate(Ap[i, :] * exp_p[i, :], As_out[i, :] * exp_s[i, :], mode='same') * np.exp(-1j * z * KK_i[i, :])
            Ap[i, :] = Ap[i, :] + common_factor_p_b[i, :] * convolve(As_out[i, :] * exp_s[i, :], Ai_out[i, :] * exp_i[i, :], mode='same') * np.exp(-1j * z * KK_p[i, :])
    else:
        for i in range(Ap.shape[0]):
            for j in range(Ap.shape[1]):
                As_out[i, j, :] = As_out[i, j, :] + common_factor_s_b[i, j, :] * correlate(Ap[i, j, :] * exp_p[i, j, :], Ai_vac[i, j, :] * exp_i[i, j, :], mode='same') * np.exp(-1j * z * KK_s[i, j, :])
                As_vac[i, j, :] = As_vac[i, j, :] + common_factor_s_b[i, j, :] * correlate(Ap[i, j, :] * exp_p[i, j, :], Ai_out[i, j, :] * exp_i[i, j, :], mode='same') * np.exp(-1j * z * KK_s[i, j, :])
                Ai_out[i, j, :] = Ai_out[i, j, :] + common_factor_i_b[i, j, :] * correlate(Ap[i, j, :] * exp_p[i, j, :], As_vac[i, j, :] * exp_s[i, j, :], mode='same') * np.exp(-1j * z * KK_i[i, j, :])
                Ai_vac[i, j, :] = Ai_vac[i, j, :] + common_factor_i_b[i, j, :] * correlate(Ap[i, j, :] * exp_p[i, j, :], As_out[i, j, :] * exp_s[i, j, :], mode='same') * np.exp(-1j * z * KK_i[i, j, :])
                Ap[i, j, :] = Ap[i, j, :] + common_factor_p_b[i, j, :] * convolve(As_out[i, j, :] * exp_s[i, j, :], Ai_out[i, j, :] * exp_i[i, j, :], mode='same') * np.exp(-1j * z * KK_p[i, j, :])

    return Ap, As_out, Ai_out, As_vac, Ai_vac

def linear_step(Ap, As_out, Ai_out, As_vac, Ai_vac, dz, KK_p, KK_s, KK_i, H_wp, H_ws, H_wi):
    Ap = np.multiply(Ap, H_wp) * np.exp(1j * KK_p * dz)
    As_out = np.multiply(As_out, H_ws) * np.exp(1j * KK_s * dz)
    Ai_out = np.multiply(Ai_out, H_wi) * np.exp(1j * KK_i * dz)
    As_vac = np.multiply(As_vac, H_ws) * np.exp(1j * KK_s * dz)
    Ai_vac = np.multiply(Ai_vac, H_wi) * np.exp(1j * KK_i * dz)
    return Ap, As_out, Ai_out, As_vac, Ai_vac

def split_step_fourier_method(z, nonlinear_orientation, Ap, As_out, Ai_out, As_vac, Ai_vac, dimensions, dz, KK_p, KK_s, KK_i, common_factor_p, common_factor_s, common_factor_i, H_wp, H_ws, H_wi):
    # Perform the nonlinear step
    Ap, As_out, Ai_out, As_vac, Ai_vac = nonlinear_step(z, Ap, As_out, Ai_out, As_vac, Ai_vac, nonlinear_orientation, KK_p, KK_s, KK_i, common_factor_p, common_factor_s, common_factor_i, dimensions)
    
    # Perform the linear step
    Ap, As_out, Ai_out, As_vac, Ai_vac = linear_step(Ap, As_out, Ai_out, As_vac, Ai_vac, dz, KK_p, KK_s, KK_i, H_wp, H_ws, H_wi)
    
    return Ap, As_out, Ai_out, As_vac, Ai_vac

def save_energy(Ap, As, Ai, dimensions, frequency_bandwidth, beam_waist):
    E_p = np.sum(np.abs(Ap) ** 2 * frequency_bandwidth)
    E_s = np.sum(np.abs(As) ** 2 * frequency_bandwidth)
    E_i = np.sum(np.abs(Ai) ** 2 * frequency_bandwidth)
    return E_p, E_s, E_i


class InteractionSPDC:
    def __init__(self, wl_central, freq_span, Z, n_w, n_z, n_p, n_s, n_i,
                 k_p, k_s, k_i, wavevector_mismatch, d_eff, domain_bounds, frequency_bandwidth,
                 beam_waist=None, dimensions=2, X=None, Y=None, n_x=None, n_y=None, wx=None, wy=None):
        self.validate_initialization(wl_central, freq_span, Z, n_w, n_z, d_eff, dimensions, X, Y, n_x, n_y, wx, wy)
        self.lambda_p_0 = wl_central
        self.omega_p_0 = (2 * np.pi * c) / self.lambda_p_0
        self.deff = d_eff
        self.domain_bounds = domain_bounds
        self.frequency_bandwidth = frequency_bandwidth
        self.beam_waist = beam_waist
        self.sell_p = n_p
        self.sell_s = n_s
        self.sell_i = n_i
        self.k_p = k_p
        self.k_s = k_s
        self.k_i = k_i
        self.dk = wavevector_mismatch
        self.W = freq_span
        self.Z = Z
        self.Nw = n_w
        self.Nz = n_z
        self.dimensions = dimensions
        self.setup_dimensions(dimensions, X, Y, n_x, n_y, wx, wy)
        self.init_grid()

    def validate_initialization(self, wl_central, freq_span, Z, n_w, n_z, d_eff, dimensions, X, Y, n_x, n_y, wx, wy):
        if wl_central <= 0 or freq_span <= 0 or Z <= 0 or n_w <= 0 or n_z <= 0 or d_eff <= 0:
            raise ValueError("All physical parameters must be positive.")
        if dimensions not in [1, 2, 3]:
            raise ValueError("Dimensions parameter must be either 1, 2, or 3.")
        if dimensions == 2 and (X is None or n_x is None or wx is None):
            raise ValueError("Not enough parameters for 2D simulation.")
        if dimensions == 3 and (X is None or Y is None or n_x is None or n_y is None or wx is None or wy is None):
            raise ValueError("Not enough parameters for 3D simulation.")

    def setup_dimensions(self, dimensions, X, Y, n_x, n_y, wx, wy):
        if dimensions == 1:
            self.X = self.Y = self.Nx = self.Ny = 0
            self.wx = self.wy = 1
        elif dimensions == 2:
            self.X, self.Nx = X, n_x
            self.Y, self.Ny = 0, 0
            self.wx, self.wy = wx, 1
        elif dimensions == 3:
            self.X, self.Y = X, Y
            self.Nx, self.Ny = n_x, n_y
            self.wx, self.wy = wx, wy

    def init_grid(self):
        self.dz = self.Z / self.Nz
        self.z = np.linspace(0, self.Z, self.Nz)
        self.w_p = np.linspace(self.omega_p_0 - self.W / 2, self.omega_p_0 + self.W / 2, self.Nw)
        self.w_s = np.linspace(self.omega_p_0/2 - self.W / 2, self.omega_p_0 / 2 + self.W / 2, self.Nw)
        self.w_i = np.linspace(self.omega_p_0/2 - self.W / 2, self.omega_p_0 / 2 + self.W / 2, self.Nw)
        self.dw = np.abs(self.w_p[1] - self.w_p[0])
        self.initialize_grids()

    def initialize_grids(self):
        if self.dimensions == 1:
            self.initialize_1d_grids()
        elif self.dimensions == 2:
            self.initialize_2d_grids()
        elif self.dimensions == 3:
            self.initialize_3d_grids()
    
    def initialize_1d_grids(self):
        self.WW_p, self.WW_s, self.WW_i = self.w_p, self.w_s, self.w_i
        self.KK_p, self.KK_s, self.KK_i  = self.k_p(self.w_p), self.k_s(self.w_s), self.k_i(self.w_i)
        self.dx, self.dy = 0, 0

    def initialize_2d_grids(self):
        self.x = np.linspace(-self.X / 2, self.X / 2, self.Nx)
        self.dx = np.abs(self.x[1] - self.x[0])
        self.dy = 0
        self.XX_s, self.WW_s = np.meshgrid(self.x, self.w_s, indexing='ij')
        self.XX_i, self.WW_i = np.meshgrid(self.x, self.w_i, indexing='ij')
        self.XX_p, self.WW_p = np.meshgrid(self.x, self.w_p, indexing='ij')
        ones_2d = np.ones((self.Nx, self.Nw))
        self.KK_p = np.multiply(ones_2d, self.k_p(self.w_p))
        self.KK_s = np.multiply(ones_2d, self.k_s(self.w_s))
        self.KK_i = np.multiply(ones_2d, self.k_i(self.w_i))

    def initialize_3d_grids(self):
        self.x = np.linspace(-self.X / 2, self.X / 2, self.Nx)
        self.y = np.linspace(-self.Y / 2, self.Y / 2, self.Ny)
        self.dx = np.abs(self.x[1] - self.x[0])
        self.dy = np.abs(self.y[1] - self.y[0])
        self.XX_s, self.YY_s, self.WW_s = np.meshgrid(self.x, self.y, self.w_s, indexing='ij')
        self.XX_i, self.YY_i, self.WW_i = np.meshgrid(self.x, self.y, self.w_i, indexing='ij')
        self.XX_p, self.YY_p, self.WW_p = np.meshgrid(self.x, self.y, self.w_p, indexing='ij')
        ones_3d = np.ones((self.Nx, self.Ny, self.Nw))
        self.KK_p = np.multiply(ones_3d, self.k_p(self.w_p))
        self.KK_s = np.multiply(ones_3d, self.k_s(self.w_s))
        self.KK_i = np.multiply(ones_3d, self.k_i(self.w_i))

    def run(self, A1, nonlinear_profile, A2=None, A3=None):
        Ap = np.array(A1, dtype=np.complex64)  # Spectral amplitude of pump
        As_out = np.zeros_like(A1, dtype=np.complex64)  # Spectral amplitude of signal
        Ai_out = np.zeros_like(A1, dtype=np.complex64)  # Spectral amplitude of idler
        As_vac = np.zeros_like(A1, dtype=np.complex64)  # Vacuum field of signal
        Ai_vac = np.zeros_like(A1, dtype=np.complex64)  # Vacuum field of idler
        Ep = np.zeros(self.Nz, dtype=np.float32)  # Energy of pump along z
        Es = np.zeros(self.Nz, dtype=np.float32)  # Energy of signal along z
        Ei = np.zeros(self.Nz, dtype=np.float32)  # Energy of idler along z

        if(A2 is None and A3 is None):
            #window = np.where(np.abs(Ap) >= np.max(Ap), 1, 0)
            #vac = np.sqrt(hbar*2*self.omega_p_0 / (2*epsilon_0) / self.Z / self.dw)
            #noise_vac  = vac * (np.random.normal(size=Ap.shape) + 1j * np.random.normal(size=Ap.shape)) / np.sqrt(2) * window
            #As_vac += noise_vac
            #Ai_vac += noise_vac
            As_vac = np.zeros_like(Ap, dtype=np.complex64)
            Ai_vac = np.zeros_like(Ap, dtype=np.complex64)
            As_vac[len(As_vac)//2] = 1
            Ai_vac[len(Ai_vac)//2] = 1
        else:
            As_vac = np.array(A2, dtype=np.complex64)
            Ai_vac = np.array(A3, dtype=np.complex64)

        As_evolution_z = np.zeros((self.Nz, *As_out.shape), dtype=np.complex64)  # Evolution of signal field
        Ai_evolution_z = np.zeros((self.Nz, *Ai_out.shape), dtype=np.complex64)  # Evolution of second harmonic field

        # Initialize Fresnel propagators
        H_wp, H_ws, H_wi = self.initialize_fresnel_propagators()

        common_factor_p = 1j * self.WW_p**2 / c**2 / self.KK_p * self.deff
        common_factor_s = 1j * self.WW_s**2 / c**2 / self.KK_s * self.deff
        common_factor_i = 1j * self.WW_i**2 / c**2 / self.KK_i * self.deff

        # Iterate over the propagation steps
        for n in tqdm(range(self.Nz), desc="Propagation steps"):
            Ap, As_out, Ai_out, As_vac, Ai_vac = split_step_fourier_method(self.z[n], nonlinear_profile[n], Ap, As_out, Ai_out, As_vac, Ai_vac, self.dimensions, self.dz, self.KK_p, self.KK_s, self.KK_i, common_factor_p, common_factor_s, common_factor_i, H_wp, H_ws, H_wi)
            Ep[n], Es[n], Ei[n] = save_energy(Ap, As_out, Ai_out, self.dimensions, self.frequency_bandwidth, self.beam_waist)
            As_evolution_z[n] = As_out
            Ai_evolution_z[n] = Ai_out

        return Ap, As_out, Ai_out, As_vac, Ai_vac, Ep, Es, Ei, As_evolution_z, Ai_evolution_z

    def initialize_fresnel_propagators(self):
        if self.dimensions == 1:
            return np.ones_like(self.WW_p, dtype=np.complex64), np.ones_like(self.WW_s, dtype=np.complex64), np.ones_like(self.WW_i, dtype=np.complex64)
        elif self.dimensions == 2:
            H_wp = np.exp(-1j * self.dz * np.square(self.XX_p / (self.Nx * self.dx**2)) / (2 * np.pi**2 / self.KK_p))
            H_ws = np.exp(-1j * self.dz * np.square(self.XX_s / (self.Nx * self.dx**2)) / (2 * np.pi**2 / self.KK_s))
            H_wi = np.exp(-1j * self.dz * np.square(self.XX_i / (self.Nx * self.dx**2)) / (2 * np.pi**2 / self.KK_i))
            return H_wp, H_ws, H_wi
        elif self.dimensions == 3:
            H_wp = np.exp(-1j * self.dz * (np.square(self.XX_p / (self.Nx * self.dx**2)) + np.square(self.YY_p / (self.Ny * self.dy**2))) / (2 * np.pi**2 / self.KK_p))
            H_ws = np.exp(-1j * self.dz * (np.square(self.XX_s / (self.Nx * self.dx**2)) + np.square(self.YY_s / (self.Ny * self.dy**2))) / (2 * np.pi**2 / self.KK_s))
            H_wi = np.exp(-1j * self.dz * (np.square(self.XX_i / (self.Nx * self.dx**2)) + np.square(self.YY_i / (self.Ny * self.dy**2))) / (2 * np.pi**2 / self.KK_i))
            return H_wp, H_ws, H_wi
