import torch
import torch.nn as nn
from typing import List, Tuple

class AeroShockSurrogate(nn.Module):
    """
    Advanced Deep Neural Network surrogate architecture specifically formulated 
    for compressible aerodynamics and Shock-Wave/Boundary-Layer Interactions (SWBLI).
    
    This model maps parameterized inputs (e.g., spatial coordinates x, y and 
    free-stream Mach number M_inf) to the full thermofluid state vector.
    It inherently enforces thermodynamic positivity constraints (rho, p, T > 0) 
    to prevent unphysical state predictions during early training epochs.
    """
    def __init__(self, layers: List[int], use_gpu: bool = True):
        """
        Initializes the compressible surrogate network.
        
        Args:
            layers: List defining the number of neurons in each layer. 
                    Example: [3, 128, 128, 128, 5] for (x, y, M_inf) -> (rho, u, v, p, T)
            use_gpu: Hardware acceleration flag.
        """
        super(AeroShockSurrogate, self).__init__()
        
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        # Mish activation is chosen over ReLU/Tanh due to its continuous non-zero 
        # higher-order derivatives, which are crucial for computing the viscous 
        # dissipation and Laplacian terms in the compressible Navier-Stokes equations.
        self.activation = nn.Mish()
        
        # Construct the hidden layers dynamically
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            linear_layer = nn.Linear(layers[i], layers[i+1])
            # Xavier/Glorot initialization for proper variance scaling in deep networks
            nn.init.xavier_normal_(linear_layer.weight)
            self.hidden_layers.append(linear_layer)
            
        # Final output layer (No activation applied yet)
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_normal_(self.output_layer.weight)
        
        # Softplus activation for thermodynamic constraints
        self.positivity_enforcer = nn.Softplus()
        
        self.to(self.device)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass defining the non-linear mapping to the state vector.
        
        Returns:
            Tuple of physical fields: (density, u_velocity, v_velocity, pressure, temperature)
        """
        x = inputs
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            
        raw_output = self.output_layer(x)
        
        # ---------------------------------------------------------
        # Physics-Informed Output Decoding & Thermodynamic Clipping
        # ---------------------------------------------------------
        # In compressible flows, density (rho), pressure (p), and temperature (T) 
        # MUST remain strictly positive. A small epsilon (1e-5) is added to prevent 
        # singularity or division by zero in the ideal gas law (p = rho * R * T).
        
        rho = self.positivity_enforcer(raw_output[:, 0:1]) + 1e-5
        u   = raw_output[:, 1:2]  # Velocity can be negative (e.g., flow separation/recirculation)
        v   = raw_output[:, 2:3]  # Velocity can be negative
        p   = self.positivity_enforcer(raw_output[:, 3:4]) + 1e-5
        T   = self.positivity_enforcer(raw_output[:, 4:5]) + 1e-5
        
        return rho, u, v, p, T

    def compute_mach_number(self, u: torch.Tensor, v: torch.Tensor, T: torch.Tensor, gamma: float = 1.4, R: float = 287.05) -> torch.Tensor:
        """
        Analytically computes the local Mach number field during inference.
        
        Args:
            u, v: Velocity components
            T: Local static temperature
            gamma: Specific heat ratio (1.4 for air)
            R: Specific gas constant (287.05 J/kg.K for air)
        """
        velocity_magnitude = torch.sqrt(u**2 + v**2 + 1e-8)
        speed_of_sound = torch.sqrt(gamma * R * T)
        return velocity_magnitude / speed_of_sound
