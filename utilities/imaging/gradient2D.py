import os

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np
import h5py

from ..forwardBackward.forward2D import forward2D, direct2DSimulationROM
from ..forwardBackward.backward2D import backward2D
from ..forwardBackward.costFunction import computeFullCostFunction, \
                                           ComputeCostFunction, \
                                           ComputeResidual

from ..tools.interpolate import linearInterpolation
from ..output import SeismicTraceOutput
from ..imaging import globals as gb
from ..tools.filter import frequencyFilter

Nfeval = 1


def gradient2DComputation(iacq, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
                          exportResidual=False, ifreq=0, modelparametrization="c", currentModelFile=None, iterfwi=None,
                          pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", computeCF=False):
    """
    Performs a partial gradient computation for all shots of a given acquisition.

    Parameters:
    -----------
        iacq : int
            Index of acquisition in listOfAcquisition
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual \
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList \
            Default is 0
        modelparametrization : str, optional
            Model parametrization for the velocity \
            Default is 'c'
        currentModelFile : str, optional
            HDF5 file containing the model file to use in the simulation
            During an FWI run, this is the model at the current iteration, \
            it is how the CPU and GPU can exchange.
            Default is None (use the model stored in the memory)
        iterfwi : int, optional
            Index of FWI iteration \
            Default is None
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    solver = gb.solver
    low = gb.minModel
    high = gb.maxModel
    freqFilter = gb.freqList[ifreq]

    solver.setModelParametrization(modelparametrization)

    if currentModelFile is not None:
        solver.updateModelfromFile(currentModelFile, low, high, comm)

        solver.reinitSolver(comm)

    print(f"iterfwi = {iterfwi}", flush=True)

    ishot = 0
    for shot in gb.listOfAcquisition[iacq].shots:

        solver.resetPressureAtReceivers(len(shot.getReceiverCoords())) # Reinitialize seismogram with correct size
        solver.computeSrcAndRcvConstants(sourcesCoords=shot.getSourceCoords()[0], receiversCoords=shot.getReceiverCoords()) # Compute source and receiver constants

        # Update source values for every shot
        solver.getsrcvalues() # Restart source values used for the forward simulation
        solver.filterSource(freqFilter) # Filter the source signal

        # set compute cost function callback
        if computeCF:
            pCostDir += f"_{iterfwi}_G"
            computeCostFunction = ComputeCostFunction( directory=pCostDir,
                                                       dt=solver.dtSeismo,
                                                       rootname="partialCost_"+shot.id
            ).computePartialCostFunction
        else:
            computeCostFunction = None

        # set residual function callback
        drootname, dformat, dext = dinfo.split(";")
        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

        print(f"Observation data file: {filename}")

        computeResidual = ComputeResidual( filename,
                                           filtering=[frequencyFilter, freqFilter, solver.dtSeismo],
                                           interpolate=[linearInterpolation, solver.maxTime, solver.dt, solver.dtSeismo]
        ).computeResidual

        # Compute shot gradient
        partial2DGradientComputation(solver, shot, comm, exportSeismicTrace, traceDirectory, gradDirectory, computeResidual,
                                     exportResidual, iterfwi=iterfwi, computeCF=computeCostFunction)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n", flush=True)

        ishot += 1
        comm.Barrier()


def gradient2DComputationROM(iacq, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".",
                             exportResidual=False, ifreq=0, modelparametrization="c", currentModelFile=None, iterfwi=None,
                             pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", computeCF=False):
    """
    Performs a partial gradient computation for all shots of a given acquisition.

    Parameters:
    -----------
        iacq : int
            Index of acquisition in listOfAcquisition
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual \
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList \
            Default is 0
        modelparametrization : str, optional
            Model parametrization for the velocity \
            Default is 'c'
        currentModelFile : str, optional
            HDF5 file containing the model file to use in the simulation
            During an FWI run, this is the model at the current iteration, \
            it is how the CPU and GPU can exchange.
            Default is None (use the model stored in the memory)
        iterfwi : int, optional
            Index of FWI iteration \
            Default is None
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    solver = gb.solver
    low = gb.minModel
    high = gb.maxModel
    freqFilter = gb.freqList[ifreq]

    solver.setModelParametrization(modelparametrization)

    if currentModelFile is not None:
        solver.updateModelfromFile(currentModelFile, low, high, comm)

    solver.reinitSolver(comm)

    solver.solverROM = False
    solver.orderFrechet = 0
    solver.orderGS = -1

    print(f"iterfwi = {iterfwi}", flush=True)

    ishot = 0
    for shot in gb.listOfAcquisition[iacq].shots:

        solver.resetPressureAtReceivers(len(shot.getReceiverCoords())) # Reinitialize seismogram with correct size
        solver.computeSrcAndRcvConstants(sourcesCoords=shot.getSourceCoords()[0], receiversCoords=shot.getReceiverCoords()) # Compute source and receiver constants

        # Update source values for every shot
        solver.getsrcvalues() # Restart source values used for the forward simulation
        solver.filterSource(freqFilter) # Filter the source signal

        # set compute cost function callback
        if computeCF:
            pCostDir += f"_{iterfwi}_G"
            computeCostFunction = ComputeCostFunction( directory=pCostDir,
                                                       dt=solver.dtSeismo,
                                                       rootname="partialCost_"+shot.id
            ).computePartialCostFunction
        else:
            computeCostFunction = None

        # set residual function callback
        drootname, dformat, dext = dinfo.split(";")
        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

        print(f"Observation data file: {filename}")

        computeResidual = ComputeResidual( filename,
                                           filtering=[frequencyFilter, freqFilter, solver.dtSeismo],
                                           interpolate=[linearInterpolation, solver.maxTime, solver.dt, solver.dtSeismo]
        ).computeResidual

        # Compute shot gradient
        partial2DGradientComputation(solver, shot, comm, exportSeismicTrace, traceDirectory, gradDirectory, computeResidual,
                                     exportResidual, iterfwi=iterfwi, computeCF=computeCostFunction)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n", flush=True)

        ishot += 1
        comm.Barrier()


def partial2DGradientComputation(solver, shot, comm, exportSeismicTrace=True, traceDirectory=".", gradDirectory="partialGradient",
                                 computeResidual=None, exportResidual=False, iterfwi=None, computeCF=None):
    """
    Performs a forward simulation for a single shot using the 2D Acoustic SEM Solver,
    followed by residual computation and backward simulation to compute the gradient.

    Workflow:
    1. Executes a forward simulation to generate synthetic seismic data.
    2. Optionally exports the simulated seismic traces to .sgy format.
    3. Computes the residual between the simulated and observed data using the provided callback.
    4. Optionally exports the residual to .sgy format.
    5. Uses the residual as a source for the backward simulation.
    6. Executes a backward simulation to compute the gradient and stores it.

    Parameters
    -----------
        solver : AcousticSolver
            AcousticSolver object used for forward and backward simulations.
        shot : Shot
            Shot object containing shot-specific information (e.g., source and receiver coordinates).
        comm : MPI_COMM
            MPI communicator for parallel processing.
        exportSeismicTrace : bool, optional
            If True, exports the simulated seismic traces to .sgy format. Default is True.
        traceDirectory : str, optional
            Directory where the seismic traces will be exported. Default is the current directory.
        gradDirectory : str, optional
            Directory where the computed gradient will be stored. Default is 'partialGradient'.
        computeResidual : callable, optional
            Callback function to compute the residual between simulated and observed data. Default is None.
        exportResidual : bool, optional
            If True, exports the computed residual to .sgy format. Default is False.
        iterfwi : int, optional
            Index of the current FWI iteration. Default is None.
        computeCF : callable, optional
            Callback function to compute the cost function. Default is None.
    """
    global Nfeval
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    outputWaveFieldInterval = int(round(solver.dtWaveField/shot.dt))

    forward2D(solver, shot, comm, outputWaveFieldInterval,"ForwardWavefields","Partial")

    simulatedData = solver.getPressureAtReceivers(comm)

    if exportSeismicTrace:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        if iterfwi is not None:
            rootname += f"_{iterfwi}"

        SeismicTraceOutput(simulatedData, format="SEGY").export(directory=traceDirectory, rootname=rootname,
                                                                  receiverCoords=shot.getReceiverCoords(),
                                                                  sourceCoords=shot.getSourceCoords()[0],
                                                                  dt=solver.dtSeismo)

    residual = computeResidual( simulatedData )

    if exportResidual:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        if iterfwi is not None:
            rootname += f"_{iterfwi}"

        SeismicTraceOutput(residual, format="SEGY").export(directory="residual", rootname=rootname,
                                                                  receiverCoords=shot.getReceiverCoords(),
                                                                  sourceCoords=shot.getSourceCoords()[0],
                                                                  dt=solver.dtSeismo)
        # Output filtered observed data - temporary
        if "DBG" in gb.mode:
            # residual = obs - simul => obs = simul + residual
            SeismicTraceOutput(residual+simulatedData, format="SEGY").export(directory="filteredObsdata",
                                      rootname=f"obsdata_shot{shot.id}_{Nfeval}",
                                      receiverCoords=shot.getReceiverCoords(),
                                      sourceCoords=shot.getSourceCoords()[0],
                                      dt=solver.dtSeismo)

    if solver.dt != solver.dtSeismo:
        # Interpolate the residual to match the solver's time step
        residual = linearInterpolation(residual, solver.maxTime, solver.dtSeismo, solver.dt)

    if rank == 0 and computeCF is not None:
        computeCF( residual )

    # Reinitialize seismogram with correct size. In this case, sources become receivers and vice versa.
    solver.resetPressureAtReceivers(len(shot.getSourceCoords()[0]))
    solver.updateSourceValue(residual)

    backward2D(solver, shot, comm, gradDirectory, outputWaveFieldInterval,"BackwardWavefields","Partial",
               False, outputWaveFieldInterval if gb.mode == "DBG+" else -1)


def computeFull2DGradient(pGradDir="partialGradient", totalCells=None, iterfwi=None):
    """
    Compute the full gradient by summing all partial gradients.
    Notice that this routine only use one process to run.
    This is to avoid IO issues and is fast enough to sum all partial gradients.

    Parameters
    ----------
        pGradDir : str
            Path to the directory containing all partial gradients.
            Default is 'partialGradient' in the current working directory.
        totalCells : int
            Total number of cells in the model.
            Default is None.
            Must be provided and different from None.
        iterfwi : int
            If running an FWI simulation, index on step of FWI.
            Default is None.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if totalCells is None:
        raise ValueError("You must provide the total number of cells in the model for the gradient computation.")

    if pGradDir is None:
        pGradDir = "partialGradient"

    if rank == 0:
        acquisition = gb.acquisition
        directory = os.fsencode(pGradDir)

        nfiles = len(acquisition.shots)
        read = np.zeros((nfiles), dtype=int)

        nf = '' if iterfwi is None else str(iterfwi)
        h5Filename = f"fullGradient{nf}.hdf5"
        h5F = h5py.File(h5Filename, "w")

        if len(list(h5F.keys())) == 0:
            h5F.create_dataset("fullGradient", data=np.zeros(totalCells), dtype='d', chunks=True, maxshape=(totalCells,))

        # Will only end after all partial gradients are read and added to the full gradient.
        while not all(read):
            if os.path.exists(pGradDir):
                file_list = os.listdir(directory)
                if len(file_list):
                    for filename in file_list:
                        filename = os.fsdecode(filename)
                        if filename.endswith("hdf5") and filename.startswith("partialGradient_ready"):
                            shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                            if not read[shotId-1]:

                                try:
                                    # Try reading the file.
                                    h5p = h5py.File(os.path.join(pGradDir, filename), 'r')

                                    # Add the partial gradient to the full gradient.
                                    h5F["fullGradient"][:] += h5p["partialGradient"][:]

                                    shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                                    read[shotId-1] = 1
                                    h5p.close()

                                    if "DBG" not in gb.mode:
                                        os.remove(os.path.join(pGradDir, filename))
                                except Exception as e:
                                    print(f"Error opening file {filename}: {e}", flush=True)
                                    # If not accessible yet or other rank reading it.
                                    pass

        h5F.close()
        if "DBG" not in gb.mode:
            os.rmdir(pGradDir)

    comm.Barrier()


def gradf2D( x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False,
            ifreq=0, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", computeFullCF=False ):
    """
    Compute the gradient

    Parameters
    -----------
        x : numpy array
            Model
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual \
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList \
            Default is 0
        model : str, optional
            Model parametrization for the velocity \
            Default is 'c'
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
        dinfo : str, optional
            Information about seismic trace file format \
            Default is 'seismoTrace_shot;5;.sgy'
        computeFullCF : bool, optional
            Compute and return the full cost function \
            Default is False

    Returns
    --------
        gradJ : numpy array
            Full gradient
        J : float, optional
            Full cost function if computeFullCF is set to True
    """
    global Nfeval
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gb.model[:] = x[:]
    # At this point the model x corresponds to the global model.
    totalCells = x.shape[0]

    modelFile = "velocityModel.hdf5"
    h = h5py.File(modelFile, "w", driver="mpio", comm=comm)

    h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(totalCells,))
    h.close()

    if Nfeval == 1 and "DBG" in gb.mode:
        with h5py.File("velocityModel0.hdf5", "w", driver="mpio", comm=comm) as hf:
            hf.create_dataset("velocity", data=x[:], dtype="d", chunks=True, maxshape=(totalCells,))

    # In client-server mode, one client will compute the full gradient
    # and all other clients will compute the partial gradients.
    if "DBG" in gb.mode:
        gradDirectory += f"_{Nfeval}"
        fullGradFilename = f"fullGradient{Nfeval}.hdf5"
        computeFull2DGradient(gradDirectory, totalCells, Nfeval)
    else:
        fullGradFilename = f"fullGradient.hdf5"
        computeFull2DGradient(gradDirectory, totalCells)

    for i in range(len(gb.listOfAcquisition)):
        gradient2DComputation(i,
                            dataDirectory,
                            gradDirectory,
                            exportSeismicTrace,
                            traceDirectory,
                            exportResidual,
                            ifreq,
                            model,
                            modelFile,
                            iterfwi=Nfeval,
                            pCostDir=pCostDir,
                            dinfo=dinfo,
                            computeCF=computeFullCF)

    # Wait for all processes to finish. Only after all
    # processes have finished, and the fulll gradient is computed,
    # that the next steps will be carried out

    h5F = h5py.File(fullGradFilename, "r", driver="mpio", comm=comm)
    gradJ = h5F["fullGradient"][:]
    h5F.close()

    # Adjust values of the gradient according to the model parametrization used.
    # COMMENT: Will only play a role when doing multiparameter
    # if model == "1/c2":
    #     gradJ = -( x * x * x / 2 ) * gradJ
    # elif model == "1/c":
    #     gradJ = - x * x * gradJ
    # elif model == "c":
    #     pass
    # else:
    #     # Different parametrizations can be added here, like if one wants to use the density.
    #     raise ValueError("Not implemented")

    # Constraining of the gradient must happen here, since it is used to compute
    # a reasonable descent direction. The actual gradient must be preserved.
    # The gradient is constrained with respect to a minimum depth of the elements.
    zind = np.arange(0,int(gb.minDepth / (gb.solver.subdomain.Ly / (gb.solver.subdomain.Iy - 1))) * gb.solver.subdomain.Ix, dtype=int)
    dirJ = np.copy(gradJ)
    dirJ[zind] = 0.0

    # Saving the direction as an hdf5 file when in debug mode
    if rank == 0 and "DBG" in gb.mode:
        h5Filename = f"directionFullGradient{Nfeval}.hdf5"
        with h5py.File(h5Filename,'w') as h5F:
            h5F.create_dataset("direction", data=dirJ)

    pCostDir += f"_{Nfeval}_G"
    Nfeval += 1

    if computeFullCF is True:
        nfiles = len( gb.acquisition.shots )

        save = True if "DBG" in gb.mode else False
        J = computeFullCostFunction( nfiles, pCostDir, sfile=f"fullCostFunction{int(gb.freqList[ifreq])}.txt", save=save )

        return gradJ, dirJ, J

    else:

        return gradJ, dirJ




def gradf2DROM( x, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=True, traceDirectory=".", exportResidual=False,
            ifreq=0, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy", computeFullCF=False ):
    """
    Compute the gradient

    Parameters
    -----------
        x : numpy array
            Model
        dataDirectory : str
            Directory containing data for residual/costFunction computation
        gradDirectory : str, optional
            Directory for the output of the partial gradients \
            Default is 'partialGradient'
        exportSeismicTrace : bool, optional
            Whether or not to export seismic trace to .sgy \
            Default is True
        traceDirectory : str, optional
            Directory where to export the seismic traces \
            Default is current directory
        exportResidual : bool, optional
            Export or not the residual \
            Default is False
        ifreq : int, optional
            Index of the frequency in gb.freqList \
            Default is 0
        model : str, optional
            Model parametrization for the velocity \
            Default is 'c'
        pCostDir : str, optional
            Partial cost function directory \
            Default is `partialCostFunction`
        dinfo : str, optional
            Information about seismic trace file format \
            Default is 'seismoTrace_shot;5;.sgy'
        computeFullCF : bool, optional
            Compute and return the full cost function \
            Default is False

    Returns
    --------
        gradJ : numpy array
            Full gradient
        J : float, optional
            Full cost function if computeFullCF is set to True
    """
    global Nfeval
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    gb.model[:] = x[:]
    # At this point the model x corresponds to the global model.
    totalCells = x.shape[0]

    modelFile = "velocityModel.hdf5"
    h = h5py.File(modelFile, "w", driver="mpio", comm=comm)

    h.create_dataset("velocity", data=x[:], dtype='d', chunks=True, maxshape=(totalCells,))
    h.close()

    if Nfeval == 1 and gb.mode == "DBG":
        with h5py.File("velocityModel0.hdf5", "w", driver="mpio", comm=comm) as hf:
            hf.create_dataset("velocity", data=x[:], dtype="d", chunks=True, maxshape=(totalCells,))

    # In client-server mode, one client will compute the full gradient
    # and all other clients will compute the partial gradients.
    if rank == 0:
        print("DBG-VMG: gradf2D - start")

    for i in range(len(gb.listOfAcquisition)):
        gradient2DComputationROM(i,
                                 dataDirectory,
                                 gradDirectory,
                                 exportSeismicTrace,
                                 traceDirectory,
                                 exportResidual,
                                 ifreq,
                                 model,
                                 modelFile,
                                 iterfwi=Nfeval,
                                 pCostDir=pCostDir,
                                 dinfo=dinfo,
                                 computeCF=computeFullCF)

    if gb.mode == "DBG":
        gradDirectory += f"_{Nfeval}"
        fullGradFilename = f"fullGradient{Nfeval}.hdf5"
        computeFull2DGradient(gradDirectory, totalCells, Nfeval)
    else:
        fullGradFilename = f"fullGradient.hdf5"
        computeFull2DGradient(gradDirectory, totalCells)


    # Wait for all processes to finish. Only after all
    # processes have finished, and the fulll gradient is computed,
    # that the next steps will be carried out

    h5F = h5py.File(fullGradFilename, "r", driver="mpio", comm=comm)
    gradJ = h5F["fullGradient"][:]
    h5F.close()

    # Adjust values of the gradient according to the model parametrization used.
    # COMMENT: Not sure why I need this here, will only play a role when doing multiparameter
    # if model == "1/c2":
    #     gradJ = -( x * x * x / 2 ) * gradJ
    # elif model == "1/c":
    #     gradJ = - x * x * gradJ
    # elif model == "c":
    #     pass
    # else:
    #     # Different parametrizations can be added here, like if one wants to use the density.
    #     raise ValueError("Not implemented")

    # Constraining of the gradient must happen here, since it is used to compute
    # a reasonable descent direction. The actual gradient must be preserved.
    # The gradient is constrained with respect to a minimum depth of the elements.
    zind = np.arange(0,int(gb.minDepth / (gb.solver.subdomain.Ly / (gb.solver.subdomain.Iy - 1))) * gb.solver.subdomain.Ix, dtype=int)
    dirJ = np.copy(gradJ)
    dirJ[zind] = 0.0

    # Saving the direction as an hdf5 file when in debug mode
    if rank == 0:
        print("DBG-VMG: gradf2D - save direction")
        h5Filename = f"directionFullGradient.hdf5"
        with h5py.File(h5Filename,'w') as h5F:
            h5F.create_dataset("direction", data=-dirJ)

    for i in range(len(gb.listOfAcquisition)):
        direct2DSimulationROM(i,
                              False,
                              traceDirectory,
                              False,
                              dataDirectory,
                              False,
                              model,
                              modelFile,
                              ifreq,
                              pCostDir,
                              dinfo,
                              solverType="Frechetsolver",
                              ordF=1,
                              ordGS=1,
                              epsilon=0.01,
                              directionFile="directionFullGradient.hdf5")

    pCostDir += f"_{Nfeval}_G"
    Nfeval += 1



    if computeFullCF is True:
        for i in range(len(gb.listOfAcquisition)):
            direct2DSimulationROM(i,
                                  True,
                                  traceDirectory,
                                  True,
                                  dataDirectory,
                                  False,
                                  model,
                                  None,
                                  ifreq,
                                  pCostDir,
                                  dinfo,
                                  solverType="ROMsolver")


        nfiles = len( gb.acquisition.shots )
        save = True if gb.mode == "DBG" else False
        J = computeFullCostFunction( nfiles, pCostDir, sfile=f"fullCostFunction{int(gb.freqList[ifreq])}.txt", save=save )
        return gradJ, dirJ, J

    else:

        return gradJ, dirJ
