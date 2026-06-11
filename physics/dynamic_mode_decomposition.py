import numpy as np
import scipy.linalg as la
from typing import Tuple, Dict

class SpectralDMD:
    """
    Advanced Dynamic Mode Decomposition (DMD) for Supersonic Viscous Flows.
    
    This module extracts spatio-temporal coherent structures from highly non-linear 
    Shock-Wave/Boundary-Layer Interactions (SWBLI). By computing the Koopman operator 
    approximations, it isolates the dominant oscillating frequencies of shockwaves 
    and separation bubbles, independent of the neural network's latent space.
    """
    
    def __init__(self, rank_truncation: int = 20):
        """
        Args:
            rank_truncation: Number of dominant singular values to retain (r).
                             Acts as a low-pass spatial filter for turbulent noise.
        """
        self.r = rank_truncation
        self.eigenvalues = None
        self.modes = None
        self.amplitudes = None

    def fit_flow_snapshots(self, X: np.ndarray, dt: float) -> Dict[str, np.ndarray]:
        """
        Executes the exact DMD algorithm via Singular Value Decomposition (SVD).
        
        Math:
            X_1 = U * Sigma * V^*
            Atilde = U^* * X_2 * V * Sigma^(-1)
        
        Args:
            X: Flow snapshot matrix of shape (Spatial_Points, Time_Steps).
               Typically represents density or pressure gradients (Schlieren).
            dt: Time step between snapshots.
            
        Returns:
            Dictionary containing continuous-time eigenvalues and spatial modes.
        """
        # 1. Split data into time-shifted matrices
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        
        # 2. Economy-size SVD on the primary snapshot matrix
        U, Sigma, Vh = la.svd(X1, full_matrices=False)
        
        # 3. Truncate to robust rank (Filtering numerical/turbulent noise)
        Ur = U[:, :self.r]
        Sigmar = np.diag(Sigma[:self.r])
        Vr = Vh[:self.r, :].conj().T
        
        # 4. Compute the reduced Koopman operator matrix (Atilde)
        Atilde = Ur.conj().T @ X2 @ Vr @ la.inv(Sigmar)
        
        # 5. Eigen-decomposition of Atilde
        Lambda, W = la.eig(Atilde)
        
        # 6. Reconstruct exact spatial DMD modes
        Phi = X2 @ Vr @ la.inv(Sigmar) @ W
        
        # 7. Convert discrete eigenvalues to continuous time (Growth rate & Frequency)
        Omega = np.log(Lambda) / dt
        
        self.eigenvalues = Omega
        self.modes = Phi
        
        # Compute optimal amplitudes using pseudo-inverse of initial state
        x0 = X1[:, 0]
        self.amplitudes = la.pinv(Phi) @ x0
        
        print(f"[DMD Analysis] Extracted {self.r} dynamic modes. Dominant frequency: {np.abs(np.imag(Omega[0]) / (2*np.pi)):.2f} Hz")
        
        return {
            'growth_rates': np.real(Omega),
            'frequencies': np.imag(Omega) / (2 * np.pi),
            'spatial_modes': Phi,
            'mode_amplitudes': self.amplitudes
        }
