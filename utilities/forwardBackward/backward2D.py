import numpy as np
from ..output.SEGYTraceOutput import SEGYTraceOutput

#from ..display import progressBar


def adjoint2DSimulation(solver, acquisition, computeSeismicTrace, traceDirectory, seisforsource, comm, startModelFile=None):
    """
    Perform an adjoint 2D simulation for seismic modeling.
    This function is used as an example to test the adjoint modeling process. It simulates the adjoint wavefield
    propagation by reversing the roles of sources and receivers. The function can also be adapted for use in
    Reverse Time Migration (RTM) workflows.
    Args:
        solver (object): The solver object responsible for 2D wavefield computations.
        acquisition (object): The acquisition object containing shot and receiver information.
        computeSeismicTrace (bool): Flag to determine whether to compute and save seismic traces.
        traceDirectory (str): Directory where seismic traces will be saved.
        seisforsource (array): Array (Nt x Nreceivers) used as the source for adjoint propagation.
        comm (MPI.Comm): MPI communicator for parallel processing.
        startModelFile (str, optional): Path to the file containing the initial velocity model. Defaults to None.
    Notes:
        - The function reinitializes the solver with the velocity model if `startModelFile` is provided.
        - For each shot in the acquisition, the function resets the pressure at receivers, updates the source
          values with the residual, and performs the backward 2D computation.
    """

    rank = comm.Get_rank()

    if startModelFile is not None:
        solver.updateModelfromFile(startModelFile, comm=comm)

        solver.reinitSolver(comm)

    # Residual must be an array based on the number of receivers used in the forward propagation (Nt, Nreceivers).
    solver.updateSourceValue(seisforsource)

    ishot = 0
    for shot in acquisition.shots:

        # Reinitialize seismogram with correct size. In this case, sources become receivers and vice versa.
        solver.resetPressureAtReceivers(len(shot.getSourceCoords()[0]))
        solver.computeSrcAndRcvConstants(sourcesCoords=shot.getSourceCoords()[0], receiversCoords=shot.getReceiverCoords()) # Compute source and receiver constants

        backward2DComputation(solver, shot, comm, computeSeismicTrace, traceDirectory)

        #Update shot flag
        shot.flag = "Done"
        if rank == 0:
            print("Shot", shot.id, "done\n")

        ishot += 1
        comm.Barrier()


def backward2DComputation(solver, shot, comm, exportSeismicTrace=False, traceDirectory="."):
    """ Backward computation

    Parameters
    ----------
        solver : AcousticSolver
            AcousticSolver object
        shot : Shot
            Contains all informations on current shot
        comm : MPI_COMM
            MPI communicators
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy.
            Default is False
        traceDirectory: string
            Directory where to export the seismic traces.
            Default is current directory
    """
    backward2D(solver, shot, comm, exportSeismicTrace=exportSeismicTrace)

    simulatedData = solver.getPressureAtReceivers(comm)

    if exportSeismicTrace:
        rootname = f"seismoTrace_backward_shot{shot.id}_{int(solver.sourceFreq)}"

        pressOut = SEGYTraceOutput(simulatedData,
                                   rootname=rootname,
                                   directory=traceDirectory)
        pressOut.export(receiverCoords=shot.getSourceCoords()[0],
                        sourceCoords=shot.getReceiverCoords(),
                        dt=solver.dtSeismo,
                        comm=comm)


def backward2D(solver, shot, comm, gradDirectory="partialGradient", updateGradInterval=-1, WaveFieldDir="BackwardWavefields", typeWavefield="Full", exportSeismicTrace=False, outputWaveField=-1):
    """Solves the backward problem using the SEM2D solver.

    Calculations are performed for a single shot.

    This method is intended for FWI, but may be used for other purposes as well.
    Gradient will only be calculated if solver.Stype corresponds to "fwi" or "inversion".

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
        updateGradInterval : int
            Interval at which the gradient is updated.
            If set to -1, no gradient will be updated.
            Default is -1
        gradDirectory : str
            Directory where to output the partial gradients.
            Default is "partialGradient"
        WaveFieldDir : str
            Output directory for wavefields
            Default is BackwardWavefields
        typeWavefield : str
            Type of wavefield to output.
            Default is "Full" for full wavefield
            Option "Partial" outputs the wavefield at each rank individually.
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy.
            Default is True
        outputWaveField : int
            Output interval for the Wavefield
            If set to -1, no wavefield will be output
            Default is -1
    """
    rank = comm.Get_rank()

    if rank == 0:
        print("\nBackward ", shot.id, flush=True)

    # Reverse time loop
    time = solver.maxTimeSim
    dt = solver.dt #solver.dtSeismo #solver.dt
    maxCycle = int(round(solver.maxTimeSim / solver.dt))
    cycle = maxCycle
    while cycle > 0:
        if rank == 0 and cycle%100 == 0:
            print(f"Backward, time = {time:.3f}s, cycle = {int(cycle):d}\n", flush=True)

        #if not cycle % outputWaveField and outputWaveField != -1 and cycle < maxCycle:
            #solver.outputWavefield(refname="P_backward_cycle"+str(cycle)+"_shot"+str(shot.id),directory=WaveFieldDir,comm=comm,typeWavefield=typeWavefield)

        # Execute one time step backward
        backward2Donestep(solver, time, exportSeismicTrace=exportSeismicTrace)

        # Update gradient
        if not cycle % updateGradInterval and updateGradInterval != -1 and cycle < maxCycle:
            solver.updateGradient(cycle, shot.id, comm, typeWavefield=typeWavefield)

        time -= dt
        cycle -= 1#*int(round(solver.dtSeismo / solver.dt))

    # Output gradient values to hdf5 (one per shot)
    if rank == 0:
        print("Outputing Gradient...", flush=True)
    solver.storeGradientInFile(shot.id, comm, gradDirectory)

    solver.resetWaveFields(resetall=True)


def backward2Donestep(solver, time, exportSeismicTrace=False):
    """
    Executes a single backward modeling step in a 2D simulation.

    This function performs one step of the backward modeling process using the
    SEM2D solver, time, and shot information. It calculates the number of
    time samples based on the given time and solver's time step, and executes
    the solver.

    Source and receiver coordinates are reversed compared to the forward
    modeling step.

    Args:
        solver: An object representing the numerical solver for the SEM 2D simulation.
                It must have an `execute` method and a `dtSeismo` attribute representing
                the time step size of the source (residual seismogram in the backward case).
        time (float): The simulation time for the backward modeling step.
        exportSeismicTrace (bool): If the seismic traces should be exported, update
                values of PressureAtReceivers. No export costs less time.
    """

    if solver.Sname in {"SEM2D", "SEM2DROM"}:
        solver.execute(timesample=int(round(time/solver.dt)), dirflag="backward")

        if exportSeismicTrace:
            solver.updatePressureAtReceivers(timesample=int(round(time / solver.dtSeismo)), flag="backward")
        else:
            pass
    else:
        raise ValueError("Unknown solver type. Only 'SEM2D' is supported.")
