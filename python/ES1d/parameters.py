#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpi4py import MPI

# ---- MPI Initialization ----
COMM = MPI.COMM_WORLD
MPI_RANK = COMM.Get_rank()
MPI_SIZE = COMM.Get_size()

# ---- Simulation parameters ----
NUM_PARTICLES = 2 ** 14    # Number of particles
DX  = 0.1                  # Grid spacing
NUM_X  = 2 ** 8            # Number of grid points
X_LENGTH = DX * NUM_X      # Length of the simulation domain
V_STDDEV = 0.5             # Std. dev. of initial velocities
DT = 0.01                  # Time-step size
NUM_T  = 256               # Number of time steps

# ---- Local variables ----
# Local number of grid points (including 2 ghost nodes)
LOCAL_NUM_X = NUM_X // MPI_SIZE + 2                 

# Start coordinate of the subdomain
X_MIN = MPI_RANK * (NUM_X // MPI_SIZE) * DX         

# End coordinate of the subdomain
X_MAX = (MPI_RANK + 1) * (NUM_X // MPI_SIZE) * DX   

