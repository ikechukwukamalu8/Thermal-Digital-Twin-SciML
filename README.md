# Physics-Informed Deep Learning Engine for Thermal Field Reconstruction

This repository demonstrates a **Hybrid Analysis and Modeling** framework leveraging **Scientific Machine Learning (SciML)** principles to map and infer continuous structural states. It implements a **Physics-Informed Neural Network (PINN)** to solve and reconstruct high-resolution, continuous thermal distributions over a one-dimensional space from heavily constrained, sparse experimental observations.

---

## Technical Problem Context

In physical systems and industrial asset monitoring, obtaining a dense field of spatial readings is often restricted by hardware limits, deployment hazards, or physical boundaries. Traditional interpolation approaches yield high variance and ignore foundational energy behaviors. 

This repository frames a solution by embedding physical principles directly into deep learning architectures. By forcing a neural network to regularize its weights against known physical constraints, continuous system profiles can be inferred accurately using minimal data capture nodes.

---

## Core SciML Mathematical Formulation

Standard data-driven neural networks lack baseline physical logic and output physically impossible structural shapes under data scarcity. This framework regularizes a deep multi-layer perceptron by optimizing a joint objective function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}}$$

1. **Empirical Objective ($\mathcal{L}_{\text{data}}$):** Maps the standard Mean Squared Error across points where physical data configurations exist:
   $$\mathcal{L}_{\text{data}} = \frac{1}{N_d}\sum_{i=1}^{N_d} |T_{\text{pred}}(x_i) - T_{\text{actual}}(x_i)|^2$$

2. **Physical Objective ($\mathcal{L}_{\text{physics}}$):** Evaluated across an interior matrix of unlabelled collocation parameters. It utilizes automatic differentiation (**Autograd**) to compute and penalize the structural residual of a steady-state 1D heat equation equipped with a non-linear internal generation source term:
   $$\frac{d^2T}{dx^2} + \pi^2 \sin(\pi x) = 0$$

---

## Implementation Structure

* **`pinn_thermal_solver.py`**: A standalone Python/PyTorch script constructing the multi-layer neural architecture, formulating the joint loss optimizer using the Adam algorithm, and handling headless image rendering.
* **`thermal_field_reconstruction.png`**: The verified visualization asset demonstrating convergence accuracy.

---

## Performance and Results

The tracking engine achieves rapid, robust convergence patterns within 1,500 tracking iterations, minimizing physical anomalies and boundary inconsistencies down to near zero ($\approx 1.3 \times 10^{-5}$):

```text
=========================================================
  INITIALIZING SCIENTIFIC MACHINE LEARNING (SciML) ENGINE 
=========================================================
Iteration    0 | Total Loss: 5.305267 | Data Loss: 0.455409
Iteration  500 | Total Loss: 0.000031 | Data Loss: 0.000000
Iteration 1000 | Total Loss: 0.000014 | Data Loss: 0.000000
Iteration 1500 | Total Loss: 0.000013 | Data Loss: 0.000000

Optimization sequence finalized. Running continuous field diagnostics...
SUCCESS: Analysis chart compiled and saved to disk.
