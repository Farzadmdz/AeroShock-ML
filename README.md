# AeroShock-ML: Deep Learning Framework for Supersonic Viscous Flows and Shock-Wave/Boundary-Layer Interactions

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/Domain-Aerospace%20%26%20Gas%20Dynamics-red)]()
[![Research](https://img.shields.io/badge/Research-Scientific%20Computing-purple)]()

---

## Abstract

**AeroShock-ML** is an open-source computational framework designed to construct high-fidelity, differentiable surrogate models for compressible aerodynamics and supersonic viscous flows. The framework specifically investigates the highly non-linear phenomena of Shock-Wave/Boundary-Layer Interaction (SWBLI), which dictates the aerodynamic heating and structural integrity of high-speed aerospace vehicles. By integrating data-driven deep learning architectures with physics-informed regularization, AeroShock-ML accelerates the prediction of severe pressure gradients, flow separation zones, and viscous dissipation profiles, mitigating the massive computational overhead typically associated with density-based solvers in Finite Volume Methods (FVM).

---

## Scientific Motivation

In the design of supersonic and hypersonic aerospace vehicles, accurate prediction of the boundary layer behavior under adverse pressure gradients induced by shock waves is critical. Traditional numerical approaches solving the full compressible Navier-Stokes equations demand extreme mesh resolution near the walls, leading to prohibitive computational costs for iterative design and optimization. AeroShock-ML provides a neural-network-based mapping from parameterized geometries and free-stream Mach numbers to the full compressible flow field $(M, p, \rho, T)$, enabling rapid aerodynamic analysis and design space exploration without sacrificing physical consistency.

---

## Governing Physics: Compressible Navier-Stokes

The framework's physical baseline relies on the conservation of mass, momentum, and energy for viscous, compressible fluids:

### Conservation Form

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}_{inviscid} = \nabla \cdot \mathbf{F}_{viscous}$$

where the state vector $\mathbf{U}$ is defined as:

$$\mathbf{U} = [\rho, \rho u, \rho v, \rho E]^T$$

### Energy Equation with Viscous Dissipation

For high-speed flows, aerodynamic heating is dominated by viscous dissipation ($\Phi$):

$$\frac{\partial (\rho E)}{\partial t} + \nabla \cdot (\rho E \mathbf{u} + p \mathbf{u}) = \nabla \cdot (k \nabla T) + \Phi$$

### Rankine-Hugoniot Jump Conditions

The framework enforces shockwave discontinuities implicitly through the dataset mapping, bounded by the Rankine-Hugoniot relations for normal and oblique shocks:

$$\rho_1 u_{n1} = \rho_2 u_{n2}$$
$$p_1 + \rho_1 u_{n1}^2 = p_2 + \rho_2 u_{n2}^2$$

---

## Core Features

* **SWBLI Surrogate Modeling:** Predicts separation bubbles and reattachment points in viscous supersonic regimes.
* **Compressible Field Reconstruction:** Maps sparse boundary conditions to continuous density ($\rho$), pressure ($p$), and Mach number ($M$) fields.
* **Data Extraction Pipelines:** Tools for importing, normalizing, and processing unstructured mesh data from traditional CFD solvers.
* **Scientific Visualization:** Integrated MATLAB and Python scripts to generate publication-ready Schlieren-style visualizations, contour maps, and boundary layer profiles.

---

## Project Structure

```text
AeroShock-ML/
├── data/
│   ├── raw_cfd_exports/
│   └── shock_normalization.py
├── models/
│   ├── compressible_surrogate.py
│   └── boundary_layer_net.py
├── visualization/
│   ├── schlieren_renderer.py
│   └── matlab_q1_plotter.m
├── scripts/
│   └── train_supersonic.py
└── README.md
