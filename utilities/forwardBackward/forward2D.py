import os

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np


from ..imaging import globals as gb
from ..output.SEGYTraceOutput import SEGYTraceOutput
from ..tools.filter import frequencyFilter
from ..tools.interpolate import linearInterpolation
from .costFunction import ComputeCostFunction, \
                          ComputeResidual


def direct2DSimulation(iacq, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=False, dataDirectory=None,
                       exportResidual=False, modelparametrization="c", currentModelFile=None, ifreq=0,
                       pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy"):
    """
    Performs a direct propagation with the 2D SEM solver.

    It is intended to be used in a FWI routine, but may also be used \
    separately for direct simulations.

    Notice that this routine makes use of the global variables defined in globals.py.

    Parameters:
    -----------
        iacq : int
            Index of Acquisition in gb.listOfAcquisition
        exportSeismicTrace : bool
            Whether or not to export the seismic traces as .sgy files \
            Default is True
        traceDirectory : string
            Directory where to export the seismic traces \
            Default is current directory
        computeCostFunction : bool
            Whether or not to compute the costFunction \
            If True, a dataDirectory must be specified
        dataDirectory : string
            Directory containing original data for residual/costFunction computation
        exportResidual : bool
            Whether or not to export the residual \
            Default is False \
            Depends on the computeCostFunction being set to True \
        modelparametrization : str
            Model parametrization to use \
            Default is "c"
        currentModelFile : str
            HDF5 file containing the model file to use in the simulation
            During an FWI run, this is the model at the current iteration, \
            it is how the CPU and GPU can exchange.
            Default is None (use the model stored in the memory)
        ifreq : int
            Index of the frequency in gb.freqList \
            Default is 0
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

    solver.getsrcvalues() # Restart source values used for the forward simulation
    solver.filterSource(freqFilter) # Filter the source signal

    ishot = 0
    print( f"{rank} ICAQ {iacq}", flush=True )
    for shot in gb.listOfAcquisition[iacq].shots:

        solver.resetPressureAtReceivers(len(shot.getReceiverCoords())) # Reinitialize seismogram with correct size
        solver.computeSrcAndRcvConstants(sourcesCoords=shot.getSourceCoords()[0], receiversCoords=shot.getReceiverCoords()) # Compute source and receiver constants

        # Initialize the cost function
        computeCostFunction = ComputeCostFunction( directory=pCostDir,
                                                   dt=solver.dtSeismo,
                                                   rootname="partialCost_"+shot.id
        ).computePartialCostFunction

        # Initialize the computeResidual function
        drootname, dformat, dext = dinfo.split(";")
        filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

        computeResidual = ComputeResidual( filename, filtering=[frequencyFilter, freqFilter, solver.dtSeismo],
                                          interpolate=[linearInterpolation, solver.maxTime, solver.dt, solver.dtSeismo] ).computeResidual

        forward2DComputation(solver, shot, comm, exportSeismicTrace, traceDirectory, computeCostFunction, computeResidual, exportResidual=exportResidual)

        # Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n", flush=True)

        ishot += 1

        comm.Barrier()


def direct2DSimulationROM(iacq, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=False, dataDirectory=None,
                          exportResidual=False, modelparametrization="c", currentModelFile=None, ifreq=0,
                          pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy",
                          solverType="SEMsolver", alpha=0, ordF=0, ordGS=-1, epsilon=1e-2, directionFile=None):
    """
    Performs a direct propagation with the 2D SEM solver.

    It is intended to be used in a FWI routine, but may also be used \
    separately for direct simulations.

    Notice that this routine makes use of the global variables defined in globals.py.

    Parameters:
    -----------
        iacq : int
            Index of Acquisition in gb.listOfAcquisition
        exportSeismicTrace : bool
            Whether or not to export the seismic traces as .sgy files \
            Default is True
        traceDirectory : string
            Directory where to export the seismic traces \
            Default is current directory
        computeCostFunction : bool
            Whether or not to compute the costFunction \
            If True, a dataDirectory must be specified
        dataDirectory : string
            Directory containing original data for residual/costFunction computation
        exportResidual : bool
            Whether or not to export the residual \
            Default is False \
            Depends on the computeCostFunction being set to True \
        modelparametrization : str
            Model parametrization to use \
            Default is "c"
        currentModelFile : str
            HDF5 file containing the model file to use in the simulation
            During an FWI run, this is the model at the current iteration, \
            it is how the CPU and GPU can exchange.
            Default is None (use the model stored in the memory)
        ifreq : int
            Index of the frequency in gb.freqList \
            Default is 0
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

    # TODO: should I do it here or inside the loop on the shots?
    if currentModelFile is not None:
        solver.updateModelfromFile(currentModelFile, low, high, comm)

        print(f"direct2DSimulation: Maximum velocity: {np.max(solver.velocity)}", flush=True)
        print(f"direct2DSimulation: Minimum velocity: {np.min(solver.velocity)}", flush=True)

        solver.reinitSolver(comm)

    solver.getsrcvalues() # Restart source values used for the forward simulation
    solver.filterSource(freqFilter) # Filter the source signal


    ishot = 0
    print( f"{rank} ICAQ {iacq}", flush=True )
    for shot in gb.listOfAcquisition[iacq].shots:

        if solverType == "ROMsolver":
            solver.shotId = shot.id
            solver.solverROM = True
            solver.alpha = alpha
            solver.setReducedModel(comm)
        elif solverType == "SEMsolver":
            solver.solverROM = False
            solver.orderFrechet = 0
            solver.orderGS = -1
        elif solverType == "Frechetsolver":
            solver.shotId = shot.id
            solver.solverROM = False
            solver.orderFrechet = ordF
            solver.orderGS = ordGS
            solver.epsilonGS = epsilon
            solver.setDirectionFrechet(directionFile, comm)

        solver.resetPressureAtReceivers(len(shot.getReceiverCoords())) # Reinitialize seismogram with correct size
        solver.computeSrcAndRcvConstants(sourcesCoords=shot.getSourceCoords()[0], receiversCoords=shot.getReceiverCoords(), comm=comm) # Compute source and receiver constants

        # Initialize the cost function
        if computeCostFunction:
            computeCostFunction = ComputeCostFunction( directory=pCostDir,
                                                       dt=solver.dtSeismo,
                                                       rootname="partialCost_"+shot.id
                                                      ).computePartialCostFunction

            # Initialize the computeResidual function
            drootname, dformat, dext = dinfo.split(";")
            filename = os.path.join(dataDirectory, drootname) + f"{int(shot.id):0{dformat}d}{dext}"

        else:
            computeCostFunction = None

        if (dataDirectory is not None and exportResidual) or (dataDirectory is not None and computeCostFunction):
            computeResidual = ComputeResidual( filename, filtering=[frequencyFilter, freqFilter, solver.dtSeismo],
                                               interpolate=[linearInterpolation, solver.maxTime, solver.dt, solver.dtSeismo] ).computeResidual
        else:
            computeResidual = None

        forward2DComputation(solver, shot, comm, exportSeismicTrace, traceDirectory, computeCostFunction, computeResidual, exportResidual=exportResidual)

        # Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n", flush=True)

        ishot += 1

        comm.Barrier()


def forward2DComputation(solver, shot, comm, exportSeismicTrace=True, traceDirectory=".", computeCostFunction=None,
                         computeResidual=None, exportResidual=False):
    """Performs forward simulation and handle exports for one shot using the 2D SEM Acoustic Solver.

    Parameters:
    -----------
        solver : AcousticSolver
            AcousticSolver object (must be AcousticSolver2D)
        shot : Shot
            Shot object
        comm : MPI_COMM
            MPI communicators
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy.
            Default is True
        traceDirectory: str
            Directory where to export the seismic traces.
            Default is current directory
        computeCostFunction : callback
            Callback to compute the cost function.
            Default is None (not computed)
        computeResidual : callback
            Callback to compute the residual.
            Default is None (not computed)
        exportResidual : bool
            Whether or not to export the residual as a .sgy file.
            Only works if computeCostFunction and computeResidual are not None.
            Default is False
    """
    rank = comm.Get_rank()

    forward2D(solver, shot, comm, exportSeismicTrace=exportSeismicTrace)

    simulatedData = solver.getPressureAtReceivers(comm)
    # Export pressure wavefield at receivers for current shot
    if exportSeismicTrace:
        rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
        pressOut = SEGYTraceOutput( simulatedData,
                                   rootname=rootname,
                                   directory=traceDirectory)
        pressOut.export(receiverCoords=shot.getReceiverCoords(),
                        sourceCoords=shot.getSourceCoords()[0],
                        dt=solver.dtSeismo,
                        comm=comm)

    # Residual and Cost Function calculation
    if computeCostFunction is not None and computeResidual is not None:
         #simulatedData = solver.getPressureAtReceivers(comm)

        residual = computeResidual( simulatedData )
        if rank == 0:
            computeCostFunction( residual )

        if exportResidual:
            rootname = f"seismoTrace_shot{shot.id}_{int(solver.sourceFreq)}"
            residualOut = SEGYTraceOutput(residual,
                                          rootname=rootname,
                                          directory="residual")
            residualOut.export(receiverCoords=shot.getReceiverCoords(),
                               sourceCoords=shot.getSourceCoords()[0],
                               dt=solver.dtSeismo,
                               comm=comm)


def forward2D(solver, shot, comm, outputWaveField=-1, WaveFieldDir="ForwardWavefields", typeWavefield="Full",exportSeismicTrace=True):
    """Solves the forward problem using the SEM2D solver.

    Calculations are performed for a single shot.

     - 2D SEM solver should be initialized and ready to use (solve.initialize()).
     - The same for the fields (solver.initFields()).
     - Pressure at receivers should have already been reset (solver.resetPressureAtReceivers())
     so its size corresponds to the number of receivers in the current shot.

    Parameters
    ----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Contains all informations on current shot
        comm : MPI_COMM
            MPI communicators
        outputWaveField : int
            Output interval for the Wavefield
            If set to -1, no wavefield will be output
            Default is -1
        WaveFieldDir : str
            Output directory for wavefields
            Default is ForwardWavefields
        typeWavefield : str
            Type of wavefield to output.
            Default is "Full" for full wavefield
            Option "Partial" outputs the wavefield at each rank individually.
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy.
            Default is True
    """
    rank = comm.Get_rank()

    if rank == 0 :
        print("\nForward ", shot.id)

    # Time loop
    time = solver.minTimeSim
    dt = solver.dt
    cycle = int(round(solver.minTimeSim/dt))
    maxCycle = int(round(solver.maxTime/dt))
    shot.flag = "In Progress"

    start = timeC.time()
    while cycle < maxCycle:
        if rank == 0 and cycle%100 == 0:
            #progressBar(count_value=cycle, total=maxCycle-1, bar_length=20, suffix=f"Forward, time = {time:.3f}s, cycle = {int(cycle):d}")
            print(f"Forward, time = {time:.3f}s, cycle = {int(cycle):d}\n", flush=True)

        if not cycle % outputWaveField and outputWaveField != -1 and cycle > 0:
            solver.outputWavefield(refname="P_cycle"+str(cycle)+"_shot"+str(shot.id),directory=WaveFieldDir,comm=comm,typeWavefield=typeWavefield)

        # Execute one time step
        forward2Donestep(solver, time, exportSeismicTrace=exportSeismicTrace)

        time += dt
        cycle += 1


    solver.resetWaveFields()
    end = timeC.time()
    print("Solver Time:", end-start)

def forward2Donestep(solver, time, exportSeismicTrace=True):
    """
    Executes a single forward modeling step in a 2D simulation.

    This function performs one step of the forward modeling process using the
    SEM2D solver, time, and shot information. It calculates the number of
    time samples based on the given time and solver's time step, and executes
    the solver.

    Args:
        solver: An object representing the numerical solver for the SEM 2D simulation.
                It must have an `execute` method and a `dt` attribute representing
                the time step size.
        time (float): The simulation time for the forward modeling step.
        shot: An object containing information about the seismic shot. It must
                have a `getSourceCoords` method that provides the source coordinates.
        exportSeismicTrace (bool): If the seismic traces should be exported, update
                values of PressureAtReceivers. No export costs less time.
    """
    comm = MPI.COMM_WORLD

    if solver.Sname in {"SEM2D", "SEM2DROM"}:
        solver.execute(timesample=int(round(time/solver.dt)), comm=comm)

    if (not (int(round(time / solver.dt)) % int(round(solver.dtSeismo / solver.dt)))) and (time >=0.0):
        solver.updatePressureAtReceivers(timesample=int(round(time / solver.dtSeismo)))

    else:
        pass
