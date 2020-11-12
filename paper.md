---
title: 'nnde: A Python package for solving differential equations using neural networks'
tags:
  - Python
  - neural networks
  - differential equations
authors:
  - name: Eric Winter
    orcid: 0000-0001-5226-2107
    affiliation: George Mason University
  - name: Bob Weigel
    affiliation: George Mason University
date: 29 October 2020
bibliography: paper.bib

# Summary

Neural networks provide a novel approach for solving the differential equations encountered in physics. The `nnde` software provides a pure-Python package for the solution of ordinary and partial differential equations of up to second order. We present results of sample runs showing the effectiveness of the software in solving the two-dimensional diffusion problem.

# Statement of need

The current standard methods for solving differential equations (Finite Element Method) and Finite Difference Method (FDM)) have long been used for magnetohydrodynamical models. However, these methods can be complicated to parallelize and can involve large storage requirements for model outputs. A neural network approach provides straightforward parallelization, due to the independent characteristics of the computational nodes in each network layer. Additionally, the trained network solution is more compact than an FDM or FEM solution, since only the network weights and biases are required. Additionally, the neural network solution is mesh-free and does not require recomputation or interpolation to retrieve the solution at a non-grid point, as is the case with FDM or FEM. Once the network is trained, computing a solution is simply a matter of a series of simple matrix multiplications, one per network layer. The trained solution is essentially an expansion of the analytical solution in a series of arbitrary differentiable basis functions, and therefore the trained solution is also differentiable, which is particularly useful when computing gradients and fluxes in models of physical phenomena.

The `nnde` package addresses a specific need - it provides a complete implementation of a standard algorithm for solving differential equations with neural networks - the trial function approach. This code can be used as a starting point for implementing other algorithms to solve ODEs and PDEs with neural networks. The code is structured in a simple fashion, with a relatively class hierarchy, and extensive internal comments to assist the user in understanding the approach. This code emphasizes readability and explicit code structure over speed, and many opportunities for code optimization remain.

# Description

`nnde` is a pure-Python package that implements a version of the trial function algorithm initially described by `@Lagaris:1998`. This software also implements a novel algorithm for incorporating the boundary conditions of the problem directly into the neural network solution.

This software was developed for solving the coupled partial differential equations of magnetohydrodynamics (MHD), but the package can be used for any ordinary or partial differential equations of up to second order. `nnde` was initially developed entirely in Python to understand the workings of the algorithm. Planned future work will incorporate the high-performance TensorFlow library to improve the speed of solutions.

As an exmaple, consider the diffusion equation in two dimensions:

\begin{equation}
  \frac {\partial G_i} {\partial p} = \frac {\partial G_i} {\partial \psi_{ti}} \frac {\partial \psi_{ti}} {\partial p} + \sum_{j=1}^m \frac {\partial G_i} {\partial \nabla_{ij} \psi_{ti}} \frac {\partial \nabla_{ij} \psi_{ti}} {\partial p} + \sum_{j=1}^m \frac {\partial G_i} {\partial \nabla_{ij}^2 \psi_{ti}} \frac {\partial \nabla_{ij}^2 \psi_{ti}} {\partial p}
  \label{diffusion2D}
\end{equation}

with all boundaries fixed at $0$, and an initial condition of:

\begin{equation}
  \psi(x,y,0) &= \sin(\pi x) \sin(\pi y)
\end{equation}

This problem has an analytical solution:

\begin{equation}
  \psi_a(\mathbf x) = e^{-2\pi^2 D t} \sin(\pi x) \sin(\pi y)
\end{equation}

The `nnde` package was used to create a neural network with a single hidden layer and 10 hidden nodes, and trained to solve this problem. The error in the trained solution is shown in \autoref{fig:diff2d_error}

![Error in solution of 2-D diffusion problem using `nnde` with 10 nodes.\label{fig:diff2d_error}](figures/diff2d_error_heatmaps.png)
and referenced from text using \autoref{fig:example}.

# Software repository

The `nnde` software is available on GitHub at https://github.com/elwinter/nnde.

A collection of example python scripts using `nnde`  is available on GitHub at https://github.com/elwinter/nnde_demos.

A collection of example Jupyter notebooks using `nnde` is available on GitHub at https://github.com/elwinter/nnde_notebooks.

# References
