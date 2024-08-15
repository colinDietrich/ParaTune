import numpy as np
from scipy.constants import c, epsilon_0
from scipy.signal import correlate, convolve
from scipy.integrate import solve_ivp
from tqdm import tqdm  # For progress bar

def SHG_NCME(z, y_flat, d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, original_shape, dimensions):
    """
    Nonlinear coupled-mode equations for second harmonic generation (SHG).
    
    Parameters:
        y_flat (ndarray): Flattened array containing real and imaginary parts of field amplitudes.
        d_eff (ndarray): Effective nonlinearity.
        KK_f (ndarray): Wavevector for the fundamental frequency.
        KK_sh (ndarray): Wavevector for the second harmonic frequency.
        common_factor_f (ndarray): Common factor for the fundamental frequency.
        common_factor_sh (ndarray): Common factor for the second harmonic frequency.
        original_shape (tuple): Original shape of the input arrays.
        dimensions (int): Number of dimensions (1, 2, or 3).
    
    Returns:
        ndarray: Flattened array containing real and imaginary parts of the updated field amplitudes.
    """
    # Reshape and split y_flat into real and imaginary parts
    y_complex = y_flat[:len(y_flat)//2] + 1j * y_flat[len(y_flat)//2:]
    y = y_complex.reshape(original_shape)
    A_f, A_sh = y[0], y[1]

    # Exponential factors for phase matching
    exp_f, exp_sh = np.exp(1j * z * KK_f), np.exp(1j * z * KK_sh)
    common_factor_f_b = common_factor_f * d_eff
    common_factor_sh_b = common_factor_sh * d_eff

    a_f = np.zeros_like(A_f, dtype=np.complex64)
    a_sh = np.zeros_like(A_sh, dtype=np.complex64)

    # Compute the nonlinear interaction based on the number of dimensions
    if dimensions == 1:
        a_f = common_factor_f_b * correlate(A_f * exp_f, A_sh * exp_sh, mode='same') * np.exp(-1j * z * KK_f)
        a_sh = common_factor_sh_b * convolve(A_f * exp_f, A_f * exp_f, mode='same') * np.exp(-1j * z * KK_sh)
    elif dimensions == 2:
        for i in range(A_f.shape[0]):
            a_f[i, :] = common_factor_f_b[i, :] * correlate(A_f[i, :] * exp_f[i, :], A_sh[i, :] * exp_sh[i, :], mode='same') * np.exp(-1j * z * KK_f[i, :])
            a_sh[i, :] = common_factor_sh_b[i, :] * convolve(A_f[i, :] * exp_f[i, :], A_f[i, :] * exp_f[i, :], mode='same') * np.exp(-1j * z * KK_sh[i, :])
    else:
        for i in range(A_f.shape[0]):
            for j in range(A_f.shape[1]):
                a_f[i, j, :] = common_factor_f_b[i, j, :] * correlate(A_f[i, j, :] * exp_f[i, j, :], A_sh[i, j, :] * exp_sh[i, j, :], mode='same') * np.exp(-1j * z * KK_f[i, j, :])
                a_sh[i, j, :] = common_factor_sh_b[i, j, :] * convolve(A_f[i, j, :] * exp_f[i, j, :], A_f[i, j, :] * exp_f[i, j, :], mode='same') * np.exp(-1j * z * KK_sh[i, j, :])

    # Flatten and separate real and imaginary parts for the solver
    a_f_real_imag = np.concatenate([np.real(a_f).flatten(), np.imag(a_f).flatten()])
    a_sh_real_imag = np.concatenate([np.real(a_sh).flatten(), np.imag(a_sh).flatten()])
    return np.concatenate([a_f_real_imag, a_sh_real_imag])

def runge_kutta_method(z0, zend, A0, d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, abs_tol, rel_tol, dimensions):
    """
    Solves the coupled-mode equations using the Runge-Kutta method.

    Parameters:
        z0 (float): Initial z-coordinate.
        zend (float): Final z-coordinate.
        A0 (ndarray): Initial field amplitudes.
        d_eff (ndarray): Effective nonlinearity.
        KK_f (ndarray): Wavevector for the fundamental frequency.
        KK_sh (ndarray): Wavevector for the second harmonic frequency.
        common_factor_f (ndarray): Common factor for the fundamental frequency.
        common_factor_sh (ndarray): Common factor for the second harmonic frequency.
        abs_tol (float): Absolute tolerance for the solver.
        rel_tol (float): Relative tolerance for the solver.
        dimensions (int): Number of dimensions (1, 2, or 3).
    
    Returns:
        ndarray: Final field amplitudes.
    """
    # Flatten the initial condition
    original_shape = A0.shape
    A0_complex_flat = A0.flatten()
    A0_real_imag_flat = np.concatenate([np.real(A0_complex_flat), np.imag(A0_complex_flat)])

    # Solve the ODE using Runge-Kutta method
    sol = solve_ivp(SHG_NCME, [z0, zend], A0_real_imag_flat, method='RK45', rtol=rel_tol, atol=abs_tol,
                    args=(d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, original_shape, dimensions))

    if sol.status == -1:
        raise RuntimeError("Integration failed.")

    # Reshape the solution back to its original shape
    y_final_complex = sol.y[:len(sol.y)//2, -1] + 1j * sol.y[len(sol.y)//2:, -1]
    return y_final_complex.reshape(original_shape)

def nonlinear_step(z0, zend, A1, A2, d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, abs_tol, rel_tol, dimensions):
    """
    Performs a nonlinear step of the split-step Fourier method.

    Parameters:
        z0 (float): Initial z-coordinate.
        zend (float): Final z-coordinate.
        A1 (ndarray): Initial field amplitude for the fundamental frequency.
        A2 (ndarray): Initial field amplitude for the second harmonic frequency.
        d_eff (ndarray): Effective nonlinearity.
        KK_f (ndarray): Wavevector for the fundamental frequency.
        KK_sh (ndarray): Wavevector for the second harmonic frequency.
        common_factor_f (ndarray): Common factor for the fundamental frequency.
        common_factor_sh (ndarray): Common factor for the second harmonic frequency.
        abs_tol (float): Absolute tolerance for the solver.
        rel_tol (float): Relative tolerance for the solver.
        dimensions (int): Number of dimensions (1, 2, or 3).

    Returns:
        tuple: Updated field amplitudes for the fundamental and second harmonic frequencies.
    """
    sol = runge_kutta_method(z0, zend, np.asarray([A1, A2]), d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, abs_tol, rel_tol, dimensions)
    return sol[0], sol[1]

def linear_step(A, dz, KK_f, KK_sh, H_wf, H_wsh):
    """
    Performs a linear step of the split-step Fourier method.

    Parameters:
        A (ndarray): Field amplitudes for the fundamental and second harmonic frequencies.
        dz (float): Step size in the z-direction.
        KK_f (ndarray): Wavevector for the fundamental frequency.
        KK_sh (ndarray): Wavevector for the second harmonic frequency.
        H_wf (ndarray): Fresnel propagator for the fundamental frequency.
        H_wsh (ndarray): Fresnel propagator for the second harmonic frequency.

    Returns:
        tuple: Updated field amplitudes for the fundamental and second harmonic frequencies.
    """
    A_f_out = np.multiply(A[0], H_wf) * np.exp(1j * KK_f * dz)
    A_sh_out = np.multiply(A[1], H_wsh) * np.exp(1j * KK_sh * dz)
    return A_f_out, A_sh_out

def split_step_fourier_method(z0, zend, d_eff, Af, Ash, dimensions, dz, KK_f, KK_sh, common_factor_f, common_factor_sh, H_wf, H_wsh, abs_tol, rel_tol):
    """
    Solves the SHG equations using the split-step Fourier method.

    Parameters:
        z0 (float): Initial z-coordinate.
        zend (float): Final z-coordinate.
        d_eff (ndarray): Effective nonlinearity.
        Af (ndarray): Initial field amplitude for the fundamental frequency.
        Ash (ndarray): Initial field amplitude for the second harmonic frequency.
        dimensions (int): Number of dimensions (1, 2, or 3).
        dz (float): Step size in the z-direction.
        KK_f (ndarray): Wavevector for the fundamental frequency.
        KK_sh (ndarray): Wavevector for the second harmonic frequency.
        common_factor_f (ndarray): Common factor for the fundamental frequency.
        common_factor_sh (ndarray): Common factor for the second harmonic frequency.
        H_wf (ndarray): Fresnel propagator for the fundamental frequency.
        H_wsh (ndarray): Fresnel propagator for the second harmonic frequency.
        abs_tol (float): Absolute tolerance for the solver.
        rel_tol (float): Relative tolerance for the solver.

    Returns:
        tuple: Updated field amplitudes for the fundamental and second harmonic frequencies.
    """
    # Perform the nonlinear step
    Af, Ash = nonlinear_step(z0, zend, Af, Ash, d_eff, KK_f, KK_sh, common_factor_f, common_factor_sh, abs_tol, rel_tol, dimensions)
    
    # Perform the linear step
    Af, Ash = linear_step(np.asarray([Af, Ash]), dz, KK_f, KK_sh, H_wf, H_wsh)
    
    return Af, Ash

def save_energy(Af, Ash, dimensions, frequency_bandwidth, beam_waist):
    """
    Computes the energy of the fields.

    Parameters:
        Af (ndarray): Field amplitude for the fundamental frequency.
        Ash (ndarray): Field amplitude for the second harmonic frequency.
        dimensions (int): Number of dimensions (1, 2, or 3).
        frequency_bandwidth (float): Frequency bandwidth.
        beam_waist (float): Beam waist.

    Returns:
        tuple: Energy of the fundamental and second harmonic fields.
    """
    E_f = np.sum(np.abs(Af) ** 2 * frequency_bandwidth)
    E_sh = np.sum(np.abs(Ash) ** 2 * frequency_bandwidth)
    return E_f, E_sh


class InteractionSHG:
    def __init__(self, wl_central, freq_span, Z, n_w, n_z, n_f, n_sh,
                 k_f, k_sh, wavevector_mismatch, d_eff, domain_bounds, frequency_bandwidth,
                 beam_waist=None, dimensions=2, X=None, Y=None, n_x=None, n_y=None, wx=None, wy=None):
        """
        Initializes the InteractionSHG class.

        Parameters:
            wl_central (float): Central wavelength.
            freq_span (float): Frequency span.
            Z (float): Interaction length.
            n_w (int): Number of frequency points.
            n_z (int): Number of z-points.
            n_f (callable): Sellmeier equation for the fundamental frequency.
            n_sh (callable): Sellmeier equation for the second harmonic frequency.
            k_f (callable): Wavevector for the fundamental frequency.
            k_sh (callable): Wavevector for the second harmonic frequency.
            wavevector_mismatch (callable): Wavevector mismatch.
            d_eff (float): Effective nonlinearity.
            domain_bounds (tuple): Bounds of the nonlinear domain.
            frequency_bandwidth (float): Frequency bandwidth.
            beam_waist (float, optional): Beam waist.
            dimensions (int, optional): Number of dimensions (1, 2, or 3). Defaults to 2.
            X (float, optional): Spatial extent in the x-direction.
            Y (float, optional): Spatial extent in the y-direction.
            n_x (int, optional): Number of points in the x-direction.
            n_y (int, optional): Number of points in the y-direction.
            wx (float, optional): Beam waist in the x-direction.
            wy (float, optional): Beam waist in the y-direction.
        """
        self.validate_initialization(wl_central, freq_span, Z, n_w, n_z, d_eff, dimensions, X, Y, n_x, n_y, wx, wy)
        self.lambda_f_0 = wl_central
        self.omega_f_0 = (2 * np.pi * c) / self.lambda_f_0
        self.deff = d_eff
        self.domain_bounds = domain_bounds
        self.frequency_bandwidth = frequency_bandwidth
        self.beam_waist = beam_waist
        self.sell_f = n_f
        self.sell_sh = n_sh
        self.k_f = k_f
        self.k_sh = k_sh
        self.dk = wavevector_mismatch
        self.W = freq_span
        self.Z = Z
        self.Nw = n_w
        self.Nz = n_z
        self.dimensions = dimensions
        self.setup_dimensions(dimensions, X, Y, n_x, n_y, wx, wy)
        self.init_grid()

    def validate_initialization(self, wl_central, freq_span, Z, n_w, n_z, d_eff, dimensions, X, Y, n_x, n_y, wx, wy):
        """
        Validates the initialization parameters.

        Parameters:
            wl_central (float): Central wavelength.
            freq_span (float): Frequency span.
            Z (float): Interaction length.
            n_w (int): Number of frequency points.
            n_z (int): Number of z-points.
            d_eff (float): Effective nonlinearity.
            dimensions (int): Number of dimensions (1, 2, or 3).
            X (float, optional): Spatial extent in the x-direction.
            Y (float, optional): Spatial extent in the y-direction.
            n_x (int, optional): Number of points in the x-direction.
            n_y (int, optional): Number of points in the y-direction.
            wx (float, optional): Beam waist in the x-direction.
            wy (float, optional): Beam waist in the y-direction.

        Raises:
            ValueError: If any of the physical parameters are not positive or if required parameters for the specified dimensions are missing.
        """
        if wl_central <= 0 or freq_span <= 0 or Z <= 0 or n_w <= 0 or n_z <= 0 or d_eff <= 0:
            raise ValueError("All physical parameters must be positive.")
        if dimensions not in [1, 2, 3]:
            raise ValueError("Dimensions parameter must be either 1, 2, or 3.")
        if dimensions == 2 and (X is None or n_x is None or wx is None):
            raise ValueError("Not enough parameters for 2D simulation.")
        if dimensions == 3 and (X is None or Y is None or n_x is None or n_y is None or wx is None or wy is None):
            raise ValueError("Not enough parameters for 3D simulation.")

    def setup_dimensions(self, dimensions, X, Y, n_x, n_y, wx, wy):
        """
        Sets up the dimensions for the simulation.

        Parameters:
            dimensions (int): Number of dimensions (1, 2, or 3).
            X (float, optional): Spatial extent in the x-direction.
            Y (float, optional): Spatial extent in the y-direction.
            n_x (int, optional): Number of points in the x-direction.
            n_y (int, optional): Number of points in the y-direction.
            wx (float, optional): Beam waist in the x-direction.
            wy (float, optional): Beam waist in the y-direction.
        """
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
        """
        Initializes the spatial and frequency grids for the simulation.
        """
        self.dz = self.Z / self.Nz
        self.z = np.linspace(0, self.Z, self.Nz)
        self.w_f = np.linspace(self.omega_f_0 - self.W / 2, self.omega_f_0 + self.W / 2, self.Nw)
        self.w_sh = np.linspace(self.omega_f_0 * 2 - self.W / 2, self.omega_f_0 * 2 + self.W / 2, self.Nw)
        self.dw = np.abs(self.w_f[1] - self.w_f[0])
        self.initialize_grids()

    def initialize_grids(self):
        """
        Initializes the spatial and frequency grids based on the number of dimensions.
        """
        if self.dimensions == 1:
            self.initialize_1d_grids()
        elif self.dimensions == 2:
            self.initialize_2d_grids()
        elif self.dimensions == 3:
            self.initialize_3d_grids()
    
    def initialize_1d_grids(self):
        """
        Initializes the spatial and frequency grids for 1D simulations.
        """
        self.WW_sh, self.WW_f = self.w_sh, self.w_f
        self.KK_f, self.KK_sh = self.k_f(self.w_f), self.k_sh(self.w_sh)
        self.dx, self.dy = 0, 0

    def initialize_2d_grids(self):
        """
        Initializes the spatial and frequency grids for 2D simulations.
        """
        self.x = np.linspace(-self.X / 2, self.X / 2, self.Nx)
        self.dx = np.abs(self.x[1] - self.x[0])
        self.dy = 0
        self.XX_sh, self.WW_sh = np.meshgrid(self.x, self.w_sh, indexing='ij')
        self.XX_f, self.WW_f = np.meshgrid(self.x, self.w_f, indexing='ij')
        ones_2d = np.ones((self.Nx, self.Nw))
        self.KK_f = np.multiply(ones_2d, self.k_f(self.w_f))
        self.KK_sh = np.multiply(ones_2d, self.k_sh(self.w_sh))

    def initialize_3d_grids(self):
        """
        Initializes the spatial and frequency grids for 3D simulations.
        """
        self.x = np.linspace(-self.X / 2, self.X / 2, self.Nx)
        self.y = np.linspace(-self.Y / 2, self.Y / 2, self.Ny)
        self.dx = np.abs(self.x[1] - self.x[0])
        self.dy = np.abs(self.y[1] - self.y[0])
        self.XX_sh, self.YY_sh, self.WW_sh = np.meshgrid(self.x, self.y, self.w_sh, indexing='ij')
        self.XX_f, self.YY_f, self.WW_f = np.meshgrid(self.x, self.y, self.w_f, indexing='ij')
        ones_3d = np.ones((self.Nx, self.Ny, self.Nw))
        self.KK_f = np.multiply(ones_3d, self.k_f(self.w_f))
        self.KK_sh = np.multiply(ones_3d, self.k_sh(self.w_sh))

    def run(self, A1, nonlinear_profile, abs_tol, rel_tol):
        """
        Runs the SHG simulation.

        Parameters:
            A1 (ndarray): Initial field amplitude for the fundamental frequency.
            nonlinear_profile (ndarray): Nonlinear profile along the propagation direction.
            abs_tol (float): Absolute tolerance for the solver.
            rel_tol (float): Relative tolerance for the solver.

        Returns:
            tuple: Final field amplitudes for the fundamental and second harmonic frequencies, energy of both fields along z, and evolution of the second harmonic field.
        """
        A1 = np.array(A1, dtype=np.complex64)  # Spectral amplitude of fundamental
        A2 = np.zeros_like(A1, dtype=np.complex64)  # Spectral amplitude of second harmonic
        E1 = np.zeros(self.Nz, dtype=np.float32)  # Energy of fundamental along z
        E2 = np.zeros(self.Nz, dtype=np.float32)  # Energy of second harmonic
        A2_evolution_z = np.zeros((self.Nz, *A2.shape), dtype=np.complex64)  # Evolution of second harmonic field

        # Initialize Fresnel propagators
        H_wf, H_wsh = self.initialize_fresnel_propagators()

        common_factor_f = 1j * self.WW_f**2 / c**2 / self.KK_f * self.deff * self.frequency_bandwidth
        common_factor_sh = 1j * self.WW_sh**2 / c**2 / self.KK_sh * self.deff * self.frequency_bandwidth

        # Iterate over the propagation steps
        for n in tqdm(range(self.Nz), desc="Propagation steps"):
            A1, A2 = split_step_fourier_method(self.z[n], self.z[n] + self.dz, nonlinear_profile[n], A1, A2, self.dimensions, self.dz, self.KK_f, self.KK_sh, common_factor_f, common_factor_sh, H_wf, H_wsh, abs_tol, rel_tol)
            E1[n], E2[n] = save_energy(A1, A2, self.dimensions, self.frequency_bandwidth, self.beam_waist)
            A2_evolution_z[n] = A2

        return A1, A2, E1, E2, A2_evolution_z

    def initialize_fresnel_propagators(self):
        """
        Initializes the Fresnel propagators for the simulation.

        Returns:
            tuple: Fresnel propagators for the fundamental and second harmonic frequencies.
        """
        if self.dimensions == 1:
            return np.ones_like(self.WW_f, dtype=np.complex64), np.ones_like(self.WW_sh, dtype=np.complex64)
        elif self.dimensions == 2:
            H_wf = np.exp(-1j * self.dz * np.square(self.XX_f / (self.Nx * self.dx**2)) / (2 * np.pi**2 / self.KK_f))
            H_wsh = np.exp(-1j * self.dz * np.square(self.XX_sh / (self.Nx * self.dx**2)) / (2 * np.pi**2 / self.KK_sh))
            return H_wf, H_wsh
        elif self.dimensions == 3:
            H_wf = np.exp(-1j * self.dz * (np.square(self.XX_f / (self.Nx * self.dx**2)) + np.square(self.YY_f / (self.Ny * self.dy**2))) / (2 * np.pi**2 / self.KK_f))
            H_wsh = np.exp(-1j * self.dz * (np.square(self.XX_sh / (self.Nx * self.dx**2)) + np.square(self.YY_sh / (self.Ny * self.dy**2))) / (2 * np.pi**2 / self.KK_sh))
            return H_wf, H_wsh
