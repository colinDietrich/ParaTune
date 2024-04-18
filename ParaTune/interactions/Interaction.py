from typing import List, Tuple, Callable
from abc import ABC
import jax.numpy as np
from scipy.constants import c
import jax.numpy as np
import jax
from functools import partial
from scipy.constants import epsilon_0, c
from jax.scipy.signal import correlate as corr
from jax.scipy.signal import fftconvolve as conv

class Interaction(ABC):

    def __init__(self, 
                 wl_central: float, 
                 freq_span: float, 
                 length: float,
                 n_w: Callable[[float], float], 
                 n_z: Callable[[float], float], 
                 n_f: Callable[[float], float], 
                 n_sh: Callable[[float], float], 
                 k_f: Callable[[float], float], 
                 k_sh: Callable[[float], float], 
                 wavevector_mismatch: Callable[[float, float], float],
                 d_eff: float, 
                 domain_bounds: np.ndarray[float], 
                 tau: float
                 ) -> None:
        """
        Initialize the Interaction object with physical and optical properties of the medium and the light interaction.

        Parameters:
        - wl_central (float): Central wavelength of the pump/fundamental field [m].
        - freq_span (float): Spectral width of the interaction [Hz].
        - length (float): Length of the medium [m].
        - n_w (Callable): Number of frequency samples function.
        - n_z (Callable): Number of spatial samples along the propagation direction.
        - n_f (Callable): Sellmeier equation for the refractive index of the fundamental field.
        - n_sh (Callable): Sellmeier equation for the refractive index of the second harmonic field.
        - k_f (Callable): Wave-vector as a function of frequency for the fundamental field.
        - k_sh (Callable): Wave-vector as a function of frequency for the second harmonic field.
        - wavevector_mismatch (Callable): Function to calculate the wavevector mismatch given two frequencies.
        - d_eff (float): Effective nonlinear coefficient of the medium [m/V].
        - domain_bounds (np.ndarray): Array specifying the positions of the boundaries of each domain within the medium.
        - tau (float): Full-width at half-maximum (FWHM) of the pulse duration [s].
        """

        self.lambda_f_0 = wl_central  # Central wavelength of the pump/fundamental field [m].
        self.omega_f_0 = (2*np.pi*c)/self.lambda_f_0  # Central angular frequency of the signal mode [rad/s].
        self.deff = d_eff  # Effective nonlinear coefficient [m/V].
        self.domain_bounds = domain_bounds  # Domain boundaries within the medium.
        self.tau = tau  # FWHM pulse duration [s].
        self.sell_f = n_f  # Sellmeier equation for fundamental field.
        self.sell_sh = n_sh  # Sellmeier equation for second harmonic field.
        self.k_f = k_f  # Wave-vector for fundamental field.
        self.k_sh = k_sh  # Wave-vector for second harmonic field.
        self.dk = wavevector_mismatch  # Wavevector mismatch function.
        self.W = freq_span  # Spectral window [Hz].
        self.Z = length  # Length of the medium [m].
        self.Nw = n_w  # Number of frequency samples.
        self.Nz = n_z  # Number of spatial samples along z-axis.
        self.tau = tau  # FWHM pulse duration [s].

        self.init_grid()  # Initialize the computational grid for the interaction.


    def init_grid(self):
        """Initialize the computational grid for the simulation."""
        self.dz = self.Z / self.Nz(self.Z)  # Spatial step along the z-axis.
        self.z = np.linspace(0, self.Z, self.Nz(self.Z))  # Spatial grid along the z-axis.

        # Frequency grids for the fundamental and second harmonic fields.
        self.w_f = np.linspace(self.omega_f_0 - self.W/2, self.omega_f_0 + self.W/2, self.Nw(self.omega_f_0))
        self.w_sh = np.linspace(self.omega_f_0*2 - self.W/2, self.omega_f_0*2 + self.W/2, self.Nw(self.omega_f_0*2))
        self.dw = self.w_f[1] - self.w_f[0]  # Frequency step.

    def SHG_NCME(self, _z: float, _A: np.ndarray, _d: float) -> np.ndarray:
        """
        Defines the nonlinear coupled mode equations for Second Harmonic Generation (SHG).

        Parameters:
        - _z (float): Current position along the propagation axis.
        - _A (np.ndarray): Array containing the complex amplitudes of the fundamental and second harmonic fields.
        - _d (float): Domain-specific parameter, usually related to the effective nonlinearity or dispersion.

        Returns:
        - np.ndarray: Derivatives of the complex amplitudes of the fundamental and second harmonic fields.
        """
        A1 = _A[0]  # Fundamental field amplitude.
        A2 = _A[1]  # Second harmonic field amplitude.

        # Compute the derivatives using the coupled mode equations.
        a1 = 1j * self.w_f / self.sell_f(self.w_f) / c * self.deff * _d * corr(A1 * np.exp(1j * _z *self.k_f(self.w_f)), A2 * np.exp(1j * _z *self.k_sh(self.w_sh)), mode='same', method='fft') * np.exp(-1j * _z *self.k_f(self.w_f))
        a2 = 1j * self.w_sh / 2 / self.sell_sh(self.w_sh) / c * self.deff * _d * conv(A1 * np.exp(1j * _z *self.k_f(self.w_f)), A1 * np.exp(1j * _z *self.k_f(self.w_f)), mode='same') * np.exp(-1j * _z *self.k_sh(self.w_sh))

        return np.array([a1, a2])

    def nonlinear_step(self, _z0: float, _d: float, _A1: np.ndarray, _A2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single nonlinear propagation step using the Runge-Kutta method
        to solve the Second Harmonic Generation (SHG) coupled mode equations.

        Parameters:
        - _z0 (float): Initial position along the propagation axis for this step.
        - _d (float): Domain-specific parameter, usually related to the effective nonlinearity or dispersion in the current segment.
        - _A1 (np.ndarray): Complex amplitude array of the fundamental field at the beginning of the step.
        - _A2 (np.ndarray): Complex amplitude array of the second harmonic field at the beginning of the step.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The updated complex amplitudes of the fundamental and second harmonic fields
        after the nonlinear step, as a tuple of two numpy arrays.
        """
        # Perform a single step of the Runge-Kutta method to solve the SHG ODEs
        sol = self.runge_kutta_method(
            _z0,  # Initial z position
            self.SHG_NCME,  # Function representing the SHG coupled mode equations
            np.asarray([_A1, _A2]),  # Initial state vector comprising the amplitudes of the fundamental and second harmonic fields
            self.dz,  # Step size along the propagation direction
            _d,  # Domain-specific parameter for the nonlinear interaction
        )

        return sol[0], sol[1]  # Return the updated amplitudes for the fundamental and second harmonic fields

    def runge_kutta_method(self, _z0: float, f: Callable, A0: np.ndarray, h: float, _d: float) -> np.ndarray:
        """
        Implements the 4th order Runge-Kutta method for advancing the solution of an ODE by a step of size h.

        Parameters:
        - _z0 (float): The initial value of the independent variable (usually spatial coordinate or time).
        - f (Callable): The function representing the ODE, dy/dz = f(z, y), where y is the dependent variable.
        - A0 (np.ndarray): The initial condition or state vector at z = _z0.
        - h (float): The step size for the independent variable (z) to advance the solution.
        - _d (float): An additional parameter passed to the ODE function, often representing physical or model parameters.

        Returns:
        - np.ndarray: The estimated value of the dependent variable (state vector) at z = _z0 + h.
        """
        # Calculate the four "slopes"
        F1 = h * f(_z0, A0, _d)  # Slope at the beginning of the interval
        F2 = h * f(_z0 + (h / 2), A0 + F1 / 2, _d)  # Slope at the midpoint, using Euler's step to F1
        F3 = h * f(_z0 + (h / 2), A0 + F2 / 2, _d)  # Slope at the midpoint, using Euler's step to F2
        F4 = h * f(_z0 + h, A0 + F3, _d)  # Slope at the end of the interval, using Euler's step to F3

        # Combine the slopes to get the final value of y at z0 + h
        y1 = A0 + (F1 + 2*F2 + 2*F3 + F4) / 6

        return y1  # Return the updated state vector

    def save_energy(self, _Af: np.ndarray, _Ash: np.ndarray) -> tuple:
        """
        Calculates and returns the energy of the fundamental and second harmonic fields.

        Parameters:
        - _Af (np.ndarray): The array representing the amplitude of the fundamental field.
        - _Ash (np.ndarray): The array representing the amplitude of the second harmonic field.

        Returns:
        - tuple: A tuple containing two elements, the energy of the fundamental field (E_f) and
                the energy of the second harmonic field (E_sh).

        Note:
        The energy calculation considers the field amplitudes, the pulse duration (tau),
        the Sellmeier equation results for fundamental (sell_f) and second harmonic (sell_sh),
        the speed of light (c), and the vacuum permittivity (epsilon_0).
        """
        # Calculate energy of the fundamental field
        E_f = np.asarray(
            np.sum(np.abs(_Af) ** 2)  # Sum of squares of the amplitude (intensity)
            * self.tau  # Multiply by pulse duration
            * (self.sell_f(self.omega_f_0) * c * epsilon_0)  # Factor from Sellmeier equation, speed of light, and permittivity
            / 2  # Divide by 2 according to the energy formula
        )

        # Calculate energy of the second harmonic field
        E_sh = np.asarray(
            np.sum(np.abs(_Ash) ** 2)  # Sum of squares of the amplitude (intensity)
            * self.tau  # Multiply by pulse duration
            * (self.sell_sh(self.omega_f_0 / 2) * c * epsilon_0)  # Factor for second harmonic using Sellmeier equation
            / 2  # Divide by 2 according to the energy formula
        )

        return E_f, E_sh  # Return the calculated energies as a tuple
    

    @partial(jax.jit, static_argnums=(0,))
    def run(self, _A1: np.ndarray, _p: np.ndarray) -> List[np.ndarray]:
        """
        Executes the nonlinear optical process simulation over the entire medium.

        This function initializes the simulation with the spectral amplitudes of the fundamental
        and second harmonic fields and iteratively computes their evolution along the propagation
        direction using the nonlinear step function. The energy of the fields is also calculated
        at each step.

        Parameters:
        - _A1 (np.ndarray): Initial spectral amplitude array of the fundamental field.
        - _p (np.ndarray): Parameter array that can vary along the propagation direction, influencing
                        the nonlinear interaction at each step.

        Returns:
        - List[np.ndarray]: A list containing the final spectral amplitudes of the fundamental and
                            second harmonic fields and their energy distributions along the propagation
                            direction.
        """
        # Initial fields and energies
        A1 = np.array(_A1, dtype=np.complex64)  # Spectral amplitude of the fundamental field
        A2 = np.array(np.zeros_like(A1), dtype=np.complex64)  # Initialize the second harmonic field as zero
        E1 = np.array(np.zeros(self.Nz), dtype=np.float32)  # Initialize the energy of the fundamental field along z as zero
        E2 = np.array(np.zeros(self.Nz), dtype=np.float32)  # Initialize the energy of the second harmonic field along z as zero
        d = _p  # Parameter array influencing the nonlinear interaction

        # Define the loop body function for JAX's fori_loop
        @partial(jax.jit, static_argnums=0)
        def body_fun(n: int, val: List[np.ndarray]) -> List[np.ndarray]:
            """
            Body function for the loop, executing a single propagation step and updating the energies.

            Parameters:
            - n (int): Current step index along the propagation direction.
            - val (List[np.ndarray]): List containing the current state of the fields and energies.

            Returns:
            - List[np.ndarray]: Updated list containing the new state of the fields and energies after the step.
            """
            # Perform a single nonlinear step
            a1, a2 = self.nonlinear_step(self.z[n], d[n], val[0], val[1])
            
            # Calculate and update energies
            val_e1, val_e2 = self.save_energy(a1, a2)
            e1, e2 = val[2], val[3]
            e1 = e1.at[n].set(val_e1)
            e2 = e2.at[n].set(val_e2)
            
            return [a1, a2, e1, e2]

        # Execute the loop over all steps along z using JAX's fori_loop for efficient execution on accelerators
        A = jax.lax.fori_loop(0, self.Nz, body_fun, [A1, A2, E1, E2])

        return A  # Return the final states of the fields and their energy distributions