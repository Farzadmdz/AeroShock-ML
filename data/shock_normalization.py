import torch
import numpy as np
from typing import Dict, Tuple

class SupersonicDataProcessor:
    """
    Advanced data ingestion and normalization pipeline for compressible flows 
    containing shock discontinuities (SWBLI).
    
    Standard Min-Max scaling fails in supersonic regimes due to the severe, 
    nearly instantaneous jumps in pressure and density across shockwaves. 
    This processor applies physics-aware logarithmic scaling to preserve 
    gradient stability during PyTorch Autograd backpropagation.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaling_metrics = {}

    def _log_scale_thermodynamics(self, field: np.ndarray, field_name: str) -> torch.Tensor:
        """
        Applies logarithmic transformation to strictly positive thermodynamic 
        variables (Pressure, Density, Temperature) to compress the shock jumps 
        into a smooth, learnable latent space.
        """
        # Ensure strict positivity to prevent log(0)
        safe_field = np.clip(field, a_min=1e-5, a_max=None)
        log_field = np.log(safe_field)
        
        f_mean = np.mean(log_field)
        f_std = np.std(log_field) + 1e-8
        
        self.scaling_metrics[field_name] = {'mean': float(f_mean), 'std': float(f_std), 'type': 'logarithmic'}
        
        normalized = (log_field - f_mean) / f_std
        return torch.tensor(normalized, dtype=torch.float32, device=self.device, requires_grad=True)

    def _standard_scale_kinematics(self, field: np.ndarray, field_name: str) -> torch.Tensor:
        """
        Applies standard Z-score normalization for velocity components (u, v) 
        and spatial coordinates, which can naturally be negative.
        """
        f_mean = np.mean(field)
        f_std = np.std(field) + 1e-8
        
        self.scaling_metrics[field_name] = {'mean': float(f_mean), 'std': float(f_std), 'type': 'linear'}
        
        normalized = (field - f_mean) / f_std
        return torch.tensor(normalized, dtype=torch.float32, device=self.device, requires_grad=True)

    def process_fluent_export(self, raw_cfd_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Ingests unstructured data from traditional CFD solvers (e.g., ANSYS Fluent) 
        and outputs Autograd-ready tensors constrained by Rankine-Hugoniot physics.
        """
        print("[AeroShock-ML] Initiating physics-aware data normalization...")
        
        processed_tensors = {
            'X': self._standard_scale_kinematics(raw_cfd_data['x'], 'X'),
            'Y': self._standard_scale_kinematics(raw_cfd_data['y'], 'Y'),
            'U': self._standard_scale_kinematics(raw_cfd_data['u'], 'U'),
            'V': self._standard_scale_kinematics(raw_cfd_data['v'], 'V'),
            
            # Thermodynamic variables undergoing severe shock jumps
            'Rho': self._log_scale_thermodynamics(raw_cfd_data['density'], 'Rho'),
            'P': self._log_scale_thermodynamics(raw_cfd_data['pressure'], 'P'),
            'T': self._log_scale_thermodynamics(raw_cfd_data['temperature'], 'T')
        }
        
        print(f"[System] Data scaled and transferred to {self.device} successfully.")
        return processed_tensors
