"""
Physics-Informed Deep Learning Engine for Thermal Field Reconstruction
Author: Ikechukwu Okechi Kamalu
Context: Standalone Executable Engine for Local IDLE / GitHub Portfolio Deployment
Description: Solves a steady-state 1D heat equation with internal generation using a PINN framework.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
# Enforce a non-interactive backend to ensure execution compatibility without a browser window
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ThermalPINN(nn.Module):
    def __init__(self):
        super(ThermalPINN, self).__init__()
        # Input: 1D spatial coordinate (x) -> Reconstructs local Temperature T(x)
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def run_production_pipeline():
    print("=========================================================")
    print("  INITIALIZING SCIENTIFIC MACHINE LEARNING (SciML) ENGINE ")
    print("=========================================================")
    
    torch.manual_seed(42)
    model = ThermalPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Formulate sparse target vectors (Simulating 5 physical hardware sensors on a thermal domain)
    x_sensors = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float32)
    T_sensors = torch.sin(np.pi * x_sensors)
    
    # Establish dense collocation spatial points for physical residual calculation
    x_collocation = torch.linspace(0, 1, 100, dtype=torch.float32).view(-1, 1)
    x_collocation.requires_grad_(True)
    
    # Optimization Loop
    epochs = 1501
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Compute Empirical Data Loss (MSE at physical sensor locations)
        T_pred_sensors = model(x_sensors)
        loss_data = nn.MSELoss()(T_pred_sensors, T_sensors)
        
        # 2. Compute Physics Loss using Automatic Differentiation (Autograd)
        # Governing PDE (Steady-state heat equation with internal heat source): d2T/dx2 + pi^2 * sin(pi * x) = 0
        T_collocation = model(x_collocation)
        
        dT_dx = torch.autograd.grad(T_collocation, x_collocation, 
                                    grad_outputs=torch.ones_like(T_collocation), 
                                    create_graph=True)[0]
        d2T_dx2 = torch.autograd.grad(dT_dx, x_collocation, 
                                      grad_outputs=torch.ones_like(dT_dx), 
                                      create_graph=True)[0]
        
        source_term = (np.pi**2) * torch.sin(np.pi * x_collocation)
        physics_residual = d2T_dx2 + source_term
        loss_physics = nn.MSELoss()(physics_residual, torch.zeros_like(physics_residual))
        
        # Composite Loss Combination
        total_loss = loss_data + 0.1 * loss_physics
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Iteration {epoch:4d} | Total Loss: {total_loss.item():.6f} | Data Loss: {loss_data.item():.6f}")
        
    print("\nOptimization sequence finalized. Running continuous field diagnostics...")
    
    # Post-optimization verification and asset rendering
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(0, 1, 100).view(-1, 1)
        T_predicted = model(x_test).numpy()
        T_exact = np.sin(np.pi * x_test.numpy())
        
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), T_exact, label='Exact Analytical Thermal Field', color='#002d5b', lw=2)
    plt.plot(x_test.numpy(), T_predicted, '--', label='PINN Reconstructed Field', color='#00d1b2', lw=2)
    plt.scatter(x_sensors.numpy(), T_sensors.numpy(), color='crimson', s=60, edgecolors='black', zorder=5, label='Sparse Physical Sensors')
    plt.title('Continuous Thermal Field Reconstruction via Physics-Informed Neural Network')
    plt.xlabel('Normalized Distance along Coordinate (x)')
    plt.ylabel('Temperature Profile T(x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Safely save the output figure asset directly to disk
    output_filename = "thermal_field_reconstruction.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("---------------------------------------------------------")
    print(f"SUCCESS: Analysis chart compiled and saved to disk: \n./{output_filename}")
    print("=========================================================")

if __name__ == "__main__":
    run_production_pipeline()
