#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpi4py import MPI
import numpy as np
from parameters import *
from pyfftw import FFTW


def deposit_charge(local_num_x: int, local_particles_x: np.ndarray, x_min: float) -> np.ndarray:
    """
    Gather the local charge density using a triangular shape function for each particle.

    Parameters
    ----------
    local_num_x : int
        Number of local grid points (including ghost nodes).
    local_particles_x : np.ndarray
        Array of local particle positions.
    x_min : float
        Start coordinate of the simulation subdomain.

    Returns
    -------
    np.ndarray
        Local charge density array (ghost nodes included).
    """
    local_rho = np.zeros(local_num_x, dtype=np.float64)

    # Loop over all local particles and deposit their charge
    for x in local_particles_x:
        x_scaled = (x - x_min) / DX
        index_x = int(x_scaled)
        weight_x_next = x_scaled - index_x
        
        # Charge is not deposited onto the left ghost node
        local_rho[index_x + 1] += 1 - weight_x_next
        local_rho[index_x + 2] += weight_x_next
    return local_rho


def poisson_solver(rho_global: np.ndarray, k_list: np.ndarray,
                   fft_obj: FFTW, ifft_obj: FFTW) -> np.ndarray:
    """
    Solve Poisson's equation in Fourier space to compute the electric field.

    Parameters
    ----------
    rho_global : np.ndarray
        Global charge density array.
    k_list : np.ndarray
        Wavenumber array for FFT.
    fft_obj : pyfftw.builders.fft
        FFT object for forward transform.
    ifft_obj : pyfftw.builders.ifft
        FFT object for inverse transform.

    Returns
    -------
    np.ndarray
        Real part of the computed electric field.
    """
    E_fft = 1j * fft_obj(rho_global) / k_list
    E_fft[0] = 0.
    
    return ifft_obj(E_fft).real


def position_update(local_particles_x: np.ndarray, local_particles_v: np.ndarray) -> np.ndarray:
    """
    Update particle positions using their velocities and the time step.

    Parameters
    ----------
    local_particles_x : np.ndarray
        Array of local particle positions.
    local_particles_v : np.ndarray
        Array of local particle velocities.

    Returns
    -------
    np.ndarray
        Updated array of local particle positions.
    """
    local_particles_x += DT * local_particles_v

    return local_particles_x


def velocity_update(local_particles_x: np.ndarray, local_particles_v: np.ndarray,
                    local_E: np.ndarray, x_min: float = X_MIN) -> np.ndarray:
    """
    Update particle velocities using the interpolated electric field.

    Parameters
    ----------
    local_particles_x : np.ndarray
        Array of local particle positions.
    local_particles_v : np.ndarray
        Array of local particle velocities.
    local_E : np.ndarray
        Local electric field array.
    x_min : float, optional
        Start coordinate of the simulation subdomain.

    Returns
    -------
    np.ndarray
        Updated array of local particle velocities.
    """
    for i, x in enumerate(local_particles_x):
        x_scaled = (x - x_min) / DX
        index_x = int(np.floor(x_scaled))
        weight_x_next = x_scaled - index_x
        local_particles_v[i] += -DT * (
            local_E[index_x + 1] * (1. - weight_x_next) +
            weight_x_next * local_E[index_x + 2]
        )

    return local_particles_v


def exchange_particles(local_particles_x: np.ndarray, local_particles_v: np.ndarray,
                       x_min: float = X_MIN, x_max: float = X_MAX, comm=MPI.COMM_WORLD,
                       mpi_rank: int = MPI_RANK, mpi_size: int = MPI_SIZE) -> tuple:
    """
    Exchange outgoing particles with neighboring domains using non-blocking MPI.

    Parameters
    ----------
    local_particles_x : np.ndarray
        Array of local particle positions.
    local_particles_v : np.ndarray
        Array of local particle velocities.
    x_min : float, optional
        Start coordinate of the simulation subdomain.
    x_max : float, optional
        End coordinate of the simulation subdomain.
    comm : MPI.Intracomm
        MPI communicator.
    mpi_rank : int, optional
        Rank of the current MPI process.
    mpi_size : int, optional
        Total number of MPI processes.

    Returns
    -------
    tuple of np.ndarray
        Updated arrays of local particle positions and velocities.
    """
    outgoing_left = local_particles_x < x_min
    outgoing_right = local_particles_x > x_max
    send_x_left = local_particles_x[outgoing_left]
    send_x_right = local_particles_x[outgoing_right]
    send_v_left = local_particles_v[outgoing_left]
    send_v_right = local_particles_v[outgoing_right]
    send_count_left = np.array([len(send_x_left)], dtype=np.int32)
    send_count_right = np.array([len(send_x_right)], dtype=np.int32)
    recv_count_left = np.empty(1, dtype=np.int32)
    recv_count_right = np.empty(1, dtype=np.int32)
    reqs = []
    # Exchange counts
    reqs.extend([
        comm.Isend(send_count_left, dest=(mpi_rank - 1) % mpi_size, tag=0),
        comm.Irecv(recv_count_right, source=(mpi_rank + 1) % mpi_size, tag=0),
        comm.Isend(send_count_right, dest=(mpi_rank + 1) % mpi_size, tag=1),
        comm.Irecv(recv_count_left, source=(mpi_rank - 1) % mpi_size, tag=1)
    ])
    MPI.Request.Waitall(reqs)
    recv_x_left = np.empty(recv_count_left[0], dtype=np.float64)
    recv_x_right = np.empty(recv_count_right[0], dtype=np.float64)
    recv_v_left = np.empty(recv_count_left[0], dtype=np.float64)
    recv_v_right = np.empty(recv_count_right[0], dtype=np.float64)
    reqs = []
    # Exchange particles
    reqs.extend([
        comm.Isend(send_x_left, dest=(mpi_rank - 1) % mpi_size, tag=2),
        comm.Irecv(recv_x_right, source=(mpi_rank + 1) % mpi_size, tag=2),
        comm.Isend(send_v_left, dest=(mpi_rank - 1) % mpi_size, tag=3),
        comm.Irecv(recv_v_right, source=(mpi_rank + 1) % mpi_size, tag=3),
        comm.Isend(send_x_right, dest=(mpi_rank + 1) % mpi_size, tag=4),
        comm.Irecv(recv_x_left, source=(mpi_rank - 1) % mpi_size, tag=4),
        comm.Isend(send_v_right, dest=(mpi_rank + 1) % mpi_size, tag=5),
        comm.Irecv(recv_v_left, source=(mpi_rank - 1) % mpi_size, tag=5)
    ])
    MPI.Request.Waitall(reqs)
    keep_mask = np.logical_not(outgoing_left | outgoing_right)
    local_particles_x = np.concatenate((local_particles_x[keep_mask], recv_x_left, recv_x_right))
    local_particles_v = np.concatenate((local_particles_v[keep_mask], recv_v_left, recv_v_right))
    local_particles_x = local_particles_x % X_LENGTH
    return local_particles_x, local_particles_v


def exchange_ghost_nodes_rho(local_rho: np.ndarray, comm=MPI.COMM_WORLD,
                             mpi_rank: int = MPI_RANK, mpi_size: int = MPI_SIZE) -> np.ndarray:
    """
    Exchange ghost node charge density with neighboring processes using non-blocking MPI.

    Parameters
    ----------
    local_rho : np.ndarray
        Local charge density array (including ghost nodes).
    comm : MPI.Intracomm
        MPI communicator.
    mpi_rank : int, optional
        Rank of the current MPI process.
    mpi_size : int, optional
        Total number of MPI processes.

    Returns
    -------
    np.ndarray
        Local charge density array with updated ghost nodes.
    """
    reqs = []
    recv_buf_left = np.zeros(1, dtype=np.float64)
    send_buf_right = np.array([local_rho[-1]], dtype=np.float64)
    # Exchange ghost nodes
    reqs.append(comm.Irecv(recv_buf_left, source=(mpi_rank - 1) % mpi_size, tag=6))
    reqs.append(comm.Isend(send_buf_right, dest=(mpi_rank + 1) % mpi_size, tag=6))
    MPI.Request.Waitall(reqs)
    local_rho[1] += recv_buf_left[0]
    return local_rho


def scatter_efield(global_E: np.ndarray, local_E: np.ndarray, local_num_x: int,
                   comm=MPI.COMM_WORLD, mpi_rank: int = MPI_RANK, mpi_size: int = MPI_SIZE) -> np.ndarray:
    """
    Scatter the global electric field array to all processes' local subdomains.

    Parameters
    ----------
    global_E : np.ndarray
        Global electric field array.
    local_E : np.ndarray
        Local electric field array to be filled.
    local_num_x : int
        Number of local grid points (including ghost nodes).
    comm : MPI.Intracomm
        MPI communicator.
    mpi_rank : int, optional
        Rank of the current MPI process.
    mpi_size : int, optional
        Total number of MPI processes.

    Returns
    -------
    np.ndarray
        Local electric field array with scattered values.
    """
    if mpi_rank == 0:
        copy_global_e = np.append(global_E, global_E[0])
        counter = 0
        global_e_with_gn = np.zeros(len(global_E) + mpi_size)
        for rank in range(mpi_size):
            global_e_with_gn[counter + rank: counter + rank + local_num_x - 1] = \
                copy_global_e[counter:counter + local_num_x - 1]
            counter += local_num_x - 2
    else:
        global_e_with_gn = None
    comm.Scatter(global_e_with_gn, local_E[1:], root=0)
    return local_E


def log_print(string: str, mpi_rank: int = MPI_RANK) -> None:
    """
    Print a message only from the root process (rank 0).

    Parameters
    ----------
    string : str
        Message to print.
    mpi_rank : int, optional
        Rank of the current MPI process.
    """
    if mpi_rank == 0:
        print(string)


def get_stats(t: int, local_particles_v: np.ndarray, global_E: np.ndarray,
              comm=MPI.COMM_WORLD, mpi_rank: int = MPI_RANK) -> None:
    """
    Print simulation statistics such as electric energy and maximum velocity.

    Parameters
    ----------
    t : int
        Current time step.
    local_particles_v : np.ndarray
        Array of local particle velocities.
    global_E : np.ndarray
        Global electric field array.
    comm : MPI.Intracomm
        MPI communicator.
    mpi_rank : int, optional
        Rank of the current MPI process.
    """
    electric_energy = np.dot(global_E, global_E)
    local_max_v = np.max(np.abs(local_particles_v))
    global_max_v = comm.reduce(local_max_v, op=MPI.MAX, root=0)
    
    if mpi_rank == 0:
        print(f"Time-steps left in simulation: {NUM_T - t}")
        print(f"Electric energy: {electric_energy:.2f}")
        print(f"Fastest particle velocity: {global_max_v:.2f}")
