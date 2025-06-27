#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
from functions import *
from parameters import *
import pyfftw

"""
Parallelized 1D electrostatic plasma simulation using MPI.

Author: Henry Holzkamp
Original code by Dr. Masatomi Iizawa

This script simulates a 1D plasma using a particle-in-cell (PIC) method,
parallelized across multiple processes with MPI for improved performance.
"""

def main():
    # Initialize particle positions and velocities per process
    np.random.seed(MPI_RANK)
    local_particles_x = np.linspace(X_MIN, X_MAX, 
                                    NUM_PARTICLES // MPI_SIZE, endpoint=False)
    local_particles_v = np.random.normal(0, V_STDDEV, NUM_PARTICLES // MPI_SIZE)

    # Initialize global and local charge density arrays
    global_rho = np.zeros(NUM_X, dtype="float64")
    local_rho = np.zeros(LOCAL_NUM_X, dtype="float64")

    # Initialize global and local electric field arrays
    global_E = np.empty(NUM_X, dtype="float64")
    local_E = np.zeros(LOCAL_NUM_X, dtype="float64")

    # fft related object creation (only on root process)
    if MPI_RANK == 0:
        if np.log2(NUM_X) % 1.0 != 0:
            raise ValueError('NUM_X needs to be a power of 2')

        k_list = (2.*np.pi) * np.fft.fftfreq(NUM_X, d=DX)
        k_list[0] = np.finfo(np.float64).eps # machine epsilon

        fft_obj = pyfftw.builders.fft(pyfftw.empty_aligned(NUM_X, dtype="float64"),
                                    planner_effort="FFTW_MEASURE")
        ifft_obj = pyfftw.builders.ifft(pyfftw.empty_aligned(NUM_X, dtype="complex128"), 
                                    planner_effort="FFTW_MEASURE")
    else:
        k_list = None
        fft_obj = None
        ifft_obj = None

    # Get start time, to measure the time taken for finishing the simulation
    start = time.time() if MPI_RANK == 0 else None


    #--------------------------MAIN LOOP--------------------------#
    # 1) Updating the local particles' positions                  #
    # 2) Exchange outgoing particles with neighboring domains     #
    # 3) Deposit the charge density (rho) onto the grid nodes     #
    # 4) Exchange the ghost nodes of rho                          #
    # 5) Gather the local rhos to get the global rho              #
    # 6) Compute the global electric field from the global rho    #
    # 7) Scatter the global electric field across the subdomains  #
    # 8) Interpolate the local eletric field and update the       #
    #  velocities                                                 #
    #-------------------------------------------------------------#

    for t in range(NUM_T): 
        log_print(f"#---------------------------------------------#")  
        log_print(f"#------------------TIME: {str(t).zfill(3)}------------------#") 
        log_print(f"#---------------------------------------------#") 
        log_print(f"Updating positions: ") 
        
        local_particles_x = position_update(local_particles_x, local_particles_v)
        log_print("Done\n") 

        log_print(f"Exchanging particles: ")
        local_particles_x, local_particles_v = exchange_particles(
            local_particles_x, local_particles_v, X_MIN, X_MAX)        
        log_print("Done\n") 

        log_print(f"Depositing charge: ") 
        local_rho = deposit_charge(LOCAL_NUM_X, local_particles_x, X_MIN)
        log_print("Done\n") 
        
        log_print(f"Exchanging ghost nodes (rho): ")
        local_rho = exchange_ghost_nodes_rho(local_rho)
        log_print("Done\n") 

        log_print(f"Gathering global rho: ") 
        COMM.Gather(local_rho[1:-1], global_rho, root=0)
        log_print("Done\n")

        log_print(f"Computing electric field (process 0): ") 
        if MPI_RANK == 0:
            global_E[:] = poisson_solver(global_rho, k_list, fft_obj, ifft_obj)
        log_print("Done\n")
        
        log_print(f"Scattering electric field: ") 
        local_E = scatter_efield(global_E, local_E, LOCAL_NUM_X)
        log_print("Done\n") 

        log_print(f"Updating velocities: ")
        local_particles_v = velocity_update(local_particles_x, local_particles_v, local_E, X_MIN)
        log_print("Done\n")

        log_print(f"#-----------Stats for t = {str(t).zfill(3)}-----------#")
        get_stats(t, local_particles_v, global_E)
        log_print("#---------------------------------------#\n")

    if MPI_RANK == 0:
        print(f"\nSimulation completed in {time.time() - start:.2f} seconds.")
        

if __name__ == '__main__':
    main()