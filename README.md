# Reservoir Computing
Reservoir computers are applied to temporal machine learning tasks where the task is to predict a time series $`\vb u(t)`$ generated from a dynamical system 
```math
    \dot{u}(t) = f_u(u(t)),
```
where the dot denotes a time derivative and $`f`$ denotes the equations of the dynamical system.  The dimension of the input system---\ie, the number of equations---is $D$.

A reservoir computer consists of three layers: an input layer $`W_{\rm in}`$, the reservoir itself, and an output layer $`\W_{\rm out}`$.  The reservoir is composed of $`N`$ nodes which can be simple nonlinear elements e.g., tanh or sigmoid activation functions, or more complex biological or physical models.  The nodes in the network are connected through an $`N \times N`$ adjacency matrix $`A`$, chosen randomly to have a connection density pnz (probability non zero) and non-zero elements uniformly chosen between $`[-1, 1]`$ and scaled such that the largest eigenvalue of $`A`$ is a number denoted the spectral radius (SR), usually $`\sim 1`$.

The input layer $\Win$ is an $`N \times D`$ dimensional matrix that maps the input signal $`u(t)`$ from $D$ dimensions into the $N$ dimensional reservoir space.  The elements of $`\W_{\rm in}`$ are chosen uniformly between $`[-1, 1]`$ and such that each row has only one non-zero element i.e., each node has input from only one dimension of $`u`$.

The output layer $\Wout$ is a matrix such that $`W_{\rm out} Q(r(t)) \equiv \hat{u}(t) \sim u(t)`$ chosen during the training phase. $`W_{\rm out}`$ is a $`D \times q N`$ dimensional matrix with $`q`$ a positive integer, $`q \in \mathcal Z^+`$, corresponding to the dimension of $`Q(r(t))`$.  $`W_{\rm out}`$ is the only part of the reservoir computer that is trained, usually through linear regression.

# Getting Started

Look at the examples "Examples_Lorenz_63.ipynb" or "Examples_Lorenz_96.ipynb" to get a feel about how to use the code.  The idea is not only to use a reservoir computer to predict chaotic time series generated from a dynamical system, but also to use algorithms from nonlinear dynamical systems to analyze and run experiments on both the reservoir and the system.  Feel free to define your own dynamical system and run your own experiments.

Ideally you won't have to touch the code in "reservoir.py," "system.py," or "lyapunov.py."  Simply define your dynamical system in f(X, t, p) format and the Jacobian df/dX(X, t, p) as well as the reservoir specifications and you are good to go.

## Packages Needed
The only non-standard package is a simple progress bar (https://github.com/verigak/progress/)

- pip install progress

# Authors
- Jason Platt (https://www.linkedin.com/in/jason-platt/)

