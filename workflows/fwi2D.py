import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

import numpy as np
import h5py

MPI.Init() # Explicit call to initialize MPI

from utilities2D.acquisition import EQUI2DAcquisition
from utilities2D.forwardBackward import func2D, func2DROM, \
                                      callback
from utilities2D.imaging import initializeGlobalVariables, \
                              setModelAsGlobal, \
                              gradf2D, gradf2DROM, \
                              Minimizer2D, SteepestDescentOptimizerROM
from utilities2D.imaging import globals as gb
from utilities2D.solvers import AcousticSolver2D, AcousticSolver2DRom
from utilities2D.input import get_model_parameters,parse_args,parse_workflow_args

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def fullWaveformInversion( minimizer, dataDirectory, gradDirectory="partialGradient", exportSeismicTrace=False, traceDirectory=".",
                          exportResidual=False, model="c", pCostDir="partialCostFunction", dinfo="seismoTrace_shot;5;.sgy" ):
    """FWI run

    Parameters:
    -----------
        minimizer : Minimizer
            Optimization method
        dataDirectory : str
            Directory for residual/costFunction computation.
        gradDirectory : str
            Directory where to output the partial gradients. Default creates (if not already exists) a new directory 'partialGradient'
        exportSeismicTrace : bool
            Whether or not to export seismic trace to .sgy. If not set default is True
        traceDirectory : str
            Directory where to export the seismic traces. If not set default is current directory
        exportResidual : bool
            Wheter or not to export the residual if computeCostFunction is set to True. Default is False
        model : str
            Model considered
        pCostDir : str
            Partial cost function directory
    """
    for ifreq, freq in enumerate(gb.freqList):
        if rank==0:
            print("\n", "*"*80, sep="")
            print(f"Filtering frequency = {freq}  [ {ifreq + 1} / {len(gb.freqList)} ]" ,flush=True)
            print("*"*80, end="\n\n")

        minimizer.optimize(x=gb.model,
                           args=(dataDirectory,
                                 gradDirectory,
                                 exportSeismicTrace,
                                 traceDirectory,
                                 exportResidual,
                                 ifreq,
                                 model,
                                 pCostDir,
                                 dinfo)
                           )

#####################################
# Main function
#####################################
if __name__ == "__main__":

    if MPI.Is_initialized():
        print("MPI initialized")
        calledinit = False
    else:
        print("MPI not initialized. Initializing...")
        MPI.Init()
        calledinit = True


    args = parse_args()

    wf_args = parse_workflow_args(args.pfile)

    mode = "DBG" if ( args.DBG or args.dbgIO ) else "REL"

    # Time Parameters
    minTime = wf_args.mintime
    maxTime = wf_args.maxtime
    dt = wf_args.dt
    dtSeismo = wf_args.dtSeismo
    dtWaveField = wf_args.dtWaveField

    # Seismic trace parameters
    traceDir = wf_args.traceDir

    if args.dbgIO:
        exportSeismicTrace = True
        exportResidual = True
    else:
        exportSeismicTrace = wf_args.exportSeismicTrace
        exportResidual = wf_args.exportResidual

    # Forward/Backward computation parameters
    sourceType = wf_args.sourceType
    sourceFreq = wf_args.sourceFreq
    if "all" not in wf_args.freqFilterList:
        freqFilterList = np.asarray(wf_args.freqFilterList.split(","), dtype=float).tolist()
    else:
        freqFilterList = ["all"]

    # FWI parameters
    method = wf_args.method
    gradDir = wf_args.gradDir
    dataDir = wf_args.dataDir
    filesToIgnore = wf_args.filesIgnore.split(",")
    dinfo = wf_args.dinfo

    model = wf_args.model

    minimizerOptions = eval(wf_args.miniOptions)

    # Set model bound constraints
    minx = minimizerOptions.pop("minModel", 500)
    maxx = minimizerOptions.pop("maxModel", 8500)

    if model == "1/c2":
        minx, maxx = 1 / maxx**2, 1 / minx**2
    elif model == "1/c":
        minx, maxx = 1 / maxx, 1 / minx

    minDepth = wf_args.minDepth

    pCostDir = wf_args.pCostDir

    taperOptions = eval(wf_args.taperOptions)

    # Set Acquisition
    acquisition = EQUI2DAcquisition(
        dt=dt,
        startFirstSourceLine=[161.031, 5.031],
        endFirstSourceLine=[2401.031, 5.031],
        startLastSourceLine=[161.031, 5.031],
        endLastSourceLine=[2401.031, 5.031],
        numberOfSourceLines=1,
        sourcesPerLine=16,
        startFirstReceiversLine=[161.013, 50.013],
        endFirstReceiversLine=[2401.013, 50.013],
        startLastReceiversLine=[161.013, 50.013],
        endLastReceiversLine=[2401.013, 50.013],
        numberOfReceiverLines=1,
        receiversPerLine=50
    )
    # FWI can be restarted from a previous model.
    if args.restart and args.rfile is not None:
        startModelFile = args.rfile
        f = h5py.File(startModelFile, 'r', driver="mpio", comm=comm)
        startModel = f["velocity"][:].flatten() # Ensure the model is 1D
        f.close()
    else:
        startModelFile = None
        # Model parametrization is done in the get_model_parameters function
        startModel, Ix, Iy = get_model_parameters(wf_args, key="velocity", comm=comm)
        # Temporary statement to scale values (because initial Marmousi is scaled)
        #startModel = startModel / (200*200) # only for 1/c2

    Ix = wf_args.Ix
    Iy = wf_args.Iy
    Lx = wf_args.Lx
    Ly = wf_args.Ly

    setModelAsGlobal(startModel)

    print(f"Rank {rank} is active\n")
    print(f"P-velocity with name {wf_args.Pmodel} model read")
    print(f"Size of the model: {startModel.shape}")
    if model == "1/c2":
        print(f"Maximum velocity: {np.sqrt(1/np.max(startModel))}")
        print(f"Minimum velocity: {np.sqrt(1/np.min(startModel))}")
    elif model == "1/c":
        print(f"Maximum velocity: {1/np.max(startModel)}")
        print(f"Minimum velocity: {1/np.min(startModel)}")
    else:
        print(f"Maximum velocity: {np.max(startModel)}")
        print(f"Minimum velocity: {np.min(startModel)}")

    solver = AcousticSolver2DRom(dt=wf_args.dt,
                                 minTime=wf_args.mintime,
                                 maxTime=wf_args.maxtime,
                                 dtSeismo=wf_args.dtSeismo,
                                 dtWaveField=wf_args.dtWaveField,
                                 sourceType=wf_args.sourceType,
                                 sourceFreq=wf_args.sourceFreq,
                                 Stype="FWI",
                                 L=[Lx,Ly],
                                 Nxy=[Ix,Iy],
                                 K=wf_args.korder,
                                 bound=wf_args.bound)

    solver.initialize(rank,comm,1/np.sqrt(startModel)) # Segment the domain and apply initial conditions (calculates the mass and stiffness matrices)
    solver.initFields() # Initialize main matrices to store the wavefield and pressure at receivers

    minimizer = SteepestDescentOptimizerROM(func2DROM,
                                            jac=gradf2DROM,
                                            bounds=[[minx,maxx]]*len(startModel),
                                            callback=callback,
                                            options=minimizerOptions)

    listOfAcquisition = acquisition.splitAcquisition()
    initializeGlobalVariables(solver, acquisition, listOfAcquisition, minx, maxx, minDepth, freqFilterList, mode)

    fullWaveformInversion(minimizer,
                          dataDir,
                          gradDirectory=gradDir,
                          exportSeismicTrace=exportSeismicTrace,
                          traceDirectory=traceDir,
                          exportResidual=exportResidual,
                          model=model,
                          pCostDir=pCostDir,
                          dinfo=dinfo)
