import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import h5py

from ..imaging import globals as gb
from .forward2D import direct2DSimulation, direct2DSimulationROM
from .costFunction import computeFullCostFunction


iter = 1
feval = 1


def func2D(x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
           exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction",
           dinfo="seismoTrace_shot;5;.sgy", computeFullCF=False ):
    """
    Callable function returning the value of the full cost function.

    This function is used in the context of FWI using a 2D SEM acoustic solver.

    Parameters
    -----------
        x : array
            data
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        exportSeismicTrace : bool
            Whether or not to export the seismic traces to a .sgy file \
            Default is True
        traceDirectory : str
            Directory where the seismic traces will be saved.
            Default is the current directory
        exportResidual : bool
            Wheter or not to export the residual.
            Default is False
        ifreq : int
            Index of the frequency in gb.freqList.
            Default to 0
        model : str
            Model parametrization to use
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`

    Returns
    --------
        J : float
            Full cost function
    """
    global feval

    comm = MPI.COMM_WORLD

    freqFilter = gb.freqList[ifreq]
    totalCells = x.shape[0]

    modelFile = "velocityModel.hdf5"
    h = h5py.File(modelFile, "w", driver="mpio", comm=comm)
    h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(totalCells,))
    h.close()

    pCostDir += f"_{feval}_F"
    for i in range(len(gb.listOfAcquisition)):
        direct2DSimulation(i, exportSeismicTrace, traceDirectory, True, dataDirectory, exportResidual, model,
                           modelFile, ifreq, pCostDir, dinfo)


    nfiles = len(gb.acquisition.shots)

    save = True if "DBG" in gb.mode else False
    J = computeFullCostFunction(nfiles, pCostDir, sfile=f"fullCostFunction{int(freqFilter)}.txt", save=save)

    print(f"\nFrequency {freqFilter} - F = {J}\n", flush=True)
    feval += 1

    return J



def func2DROM(x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
              exportResidual=False, ifreq=0, model="c", pCostDir="partialCostFunction",
              dinfo="seismoTrace_shot;5;.sgy", computeFullCF=False, alpha=None ):
    """
    Callable function returning the value of the full cost function.

    This function is used in the context of FWI using a 2D SEM acoustic solver.

    Parameters
    -----------
        x : array
            data
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        exportSeismicTrace : bool
            Whether or not to export the seismic traces to a .sgy file \
            Default is True
        traceDirectory : str
            Directory where the seismic traces will be saved.
            Default is the current directory
        exportResidual : bool
            Wheter or not to export the residual.
            Default is False
        ifreq : int
            Index of the frequency in gb.freqList.
            Default to 0
        model : str
            Model parametrization to use
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`

    Returns
    --------
        J : float
            Full cost function
    """
    global feval

    comm = MPI.COMM_WORLD

    if alpha is not None:
        solverType = "ROMsolver"
    else:
        alpha = 0
        solverType = "Frechetsolver"

    freqFilter = gb.freqList[ifreq]
    """
    totalCells = x.shape[0]

    modelFile = "velocityModel.hdf5"
    h = h5py.File(modelFile, "w", driver="mpio", comm=comm)
    h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(totalCells,))
    h.close()
    """
    pCostDir += f"_{feval}_F"
    for i in range(len(gb.listOfAcquisition)):
        direct2DSimulationROM(i,
                              exportSeismicTrace,
                              traceDirectory,
                              True,
                              dataDirectory,
                              exportResidual,
                              model,
                              None,
                              ifreq,
                              pCostDir,
                              dinfo,
                              solverType,
                              alpha)


    nfiles = len(gb.acquisition.shots)

    save = True if gb.mode == "DBG" else False
    J = computeFullCostFunction(nfiles, pCostDir, sfile=f"fullCostFunction{int(freqFilter)}.txt", save=save)

    print(f"\nFrequency {freqFilter} - F = {J}\n", flush=True)
    feval += 1

    return J



def callback(x):
    """
    Create a HDF5 file for the velocity model at the end of each iteration

    Parameters
    -----------
        x : array
            Velocity model
    """
    global iter

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    modelFile = "velocityModel" + str(iter) + ".hdf5"
    model = h5py.File(modelFile, "w", driver="mpio", comm=comm)
    model.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(x.shape[0],))
    model.close()

    print(f"callback: Maximum velocity: {max(x)}", flush=True)
    print(f"callback: Minimum velocity: {min(x)}", flush=True)

    if rank == 0:
        print( "*"*80, f"Velocity model {iter} saved as `{modelFile}`", "*"*80, sep="\n"*2, flush=True )

    iter += 1

    comm.Barrier()
