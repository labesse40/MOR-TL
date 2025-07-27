import sys
import argparse
import numpy as np

import h5py


def parse_args():
    parser = argparse.ArgumentParser("Direct and FWI workflow parser for the 2D SEM acoustic solver")
    parser.add_argument("--param_file", type=str, dest="pfile",
                        help="File containing all the parameters needed")
    parser.add_argument("--r", "--restart", action="store_true", default=False,
                        dest="restart",
                        help="Flag to restart from existing model file")
    parser.add_argument("--rfile", "--restart_model_file",
                        type=str, dest="rfile", default=None,
                        help="HDF5 file containing the model for restart. \
                                WARNING: the restart model (c, 1/c or 1/c2) used should be \
                                the same as the one requested for the current run")
    parser.add_argument("--dbg", dest="DBG",
                        action="store_true", default=False,
                        help="Debug mode\
                              (1) Save all intermediate full gradient. \
                              (2) Save all intermediate full cost function. \
                              (3) Keep all partial cost functions files. \
                              (4) Keep all partial gradient directories. \
                              (5) Saves all seismic traces. \
                              (6) Saves all residuals. \
                              (7) Saves all forward wavefields. \
                              Default is False")
    parser.add_argument("--dbg+", dest="dbgIO",
                        action="store_true", default=False,
                        help="Mode Debug+, \
                              (1) Activates the debug mode. \
                              (2) Saves all backward wavefields. \
                              (3) Saves all filtered observed seismic traces. \
                              Default is False.")

    args, _ = parser.parse_known_args()
    return args

def parse_workflow_args(pfile):
    """
    Read an input parameter file, check the values, and parse valid arguments.
    Current version allows different types of parameter files.
    :TODO detailed description of inputs/output
    """
    with open(pfile, "r") as f:
        hdrStr = f.read()

    hdrList = []
    for fl in hdrStr.split('\n'):
        l = fl.split("#")[0]
        if l:
            # add "--" to facilitate parsing that follows
            l = "--" + l
            hdrList += l.split("=",1)

    parser = argparse.ArgumentParser("FWI workflow parser")
    parser.add_argument("--Pmodel", type=str, required=False, dest="Pmodel", default=None,
                        help="Input P-velocity model")
    parser.add_argument("--NewParfile", type=bool,
                        dest="NewParfile", default=False,
                        help="Set to True if you want to use a parameter file \
                            with a format different than that used in makutu")
    parser.add_argument("--Nxs", type=float,
                        dest="Nxs", default=1,
                        help="Number of sources along x")
    parser.add_argument("--Nys", type=float,
                        dest="Nys", default=1,
                        help="Number of sources along y")
    parser.add_argument("--xs", type=float,
                        dest="xs", default=0.5,
                        help="x pos source")
    parser.add_argument("--ys", type=float,
                        dest="ys", default=0.5,
                        help="y pos source")
    parser.add_argument("--xsmin", type=float,
                        dest="xsmin", default=0.5,
                        help="x min pos source")
    parser.add_argument("--ysmin", type=float,
                        dest="ysmin", default=0.5,
                        help="y min pos source")
    parser.add_argument("--xsmax", type=float,
                        dest="xsmax", default=0.5,
                        help="x max pos source")
    parser.add_argument("--ysmax", type=float,
                        dest="ysmax", default=0.5,
                        help="y max pos source")
    parser.add_argument("--sf", type=float,
                        dest="sf", default=5,
                        help="Ricker central frequency")
    parser.add_argument("--time", type=float,
                        dest="time", default=2.0,
                        help="simulation time")
    parser.add_argument("--korder", type=int,
                        dest="korder", default=1,
                        help="FE order")
    parser.add_argument("--dsize", type=float,
                        dest="dsize", default=50.0,
                        help="domain size (meters) along x and y directions")
    parser.add_argument("--Ly", type=float,
                        dest="Ly", default=500.0,
                        help="domain size (meters) along y direction")
    parser.add_argument("--Lx", type=float,
                        dest="Lx", default=500.0,
                        help="domain size (meters) along x direction")
    parser.add_argument("--nbelem", type=int,
                        dest="nbelem", default=50,
                        help="number of elements in one direction")
    parser.add_argument("--Ix", type=int,
                        dest="Ix", default=None,
                        help="number of elements along x direction")
    parser.add_argument("--Iy", type=int,
                        dest="Iy", default=None,
                        help="number of elements along y direction")
    parser.add_argument("--dt", dest="dt",
                        default=None, type=float,
                        help="Time step of simulation")
    parser.add_argument("--vel", type=float,
                        dest="vel", default=10,
                        help="velocity")
    parser.add_argument("--bound", type=str,
                        dest="bound", default="neumann",
                        help="boundary conditions Neumann/Dirichlet/abc")
    parser.add_argument("--freesurface", dest="freesurface",
                        action="store_true", default=False,
                        help="free surface condition in upper boundary \
                            Default is false")
    parser.add_argument("--xmin", type=float,
                        dest="xmin", default=0,
                        help="minimum x offset")
    parser.add_argument("--xmax", type=float,
                        dest="xmax", default=0,
                        help="maximum x offset")
    parser.add_argument("--ymin", type=float,
                        dest="ymin", default=0,
                        help="minimum y offset")
    parser.add_argument("--ymax", type=float,
                        dest="ymax", default=0,
                        help="maximum y offset")
    parser.add_argument("--nxr", type=int,
                        dest="nxr", default=1,
                        help="number of x offsets (integer bigger or equal 1)")
    parser.add_argument("--nyr", type=int,
                        dest="nyr", default=1,
                        help="number of y offsets (integer bigger or equal 1)")
    parser.add_argument("--sref", type=str,
                        dest="sref", default=None,
                        help="Reference seismogram (observed data)")
    ##### Below the makutu specific arguments
    parser.add_argument("--mintime", dest="mintime",
                        default=None, type=float,
                        help="Min time for the simulation")
    parser.add_argument("--maxtime", dest="maxtime",
                        default=None, type=float,
                        help="Max time for the simulation")
    parser.add_argument("--dtSeismo", dest="dtSeismo",
                        default=None, type=float,
                        help="Time step for ")
    parser.add_argument("--dtWaveField", dest="dtWaveField",
                        required=True, type=float,
                        help="Time step for the wavefields. \
                         Required for a FWI calculation")
    parser.add_argument("--dataDirectory", dest="dataDir",
                        type=str, default="seismicTrace",
                        help="Directory containing data for residual/cost function" )
    parser.add_argument("--ignore_files", dest="filesIgnore",
                        type=str, default="",
                        help="Files to ignore in data directory")
    parser.add_argument("--dinfo", dest="dinfo",
                        type=str, default="seismoTrace_shot;5;.sgy",
                        help="Rootname of data traces, shot id formatting (e.g. 000123 is 6), extension, separated by `;`\
                        Default is `seismoTrace_shot;5;.sgy`")
    parser.add_argument("--trace_directory", dest="traceDir",
                        type=str, default="seismicTrace",
                        help="Seismo trace output directory")
    parser.add_argument("--exportSeismicTrace", dest="exportSeismicTrace",
                        action="store_true", default=False,
                        help="Flag to export the seismo traces. \
                            Default is false")
    parser.add_argument("--exportResidual", dest="exportResidual",
                        action="store_true", default=False,
                        help="Flag to export the residual. \
                            Default is false")
    parser.add_argument("--pCostdir", dest="pCostDir",
                        type=str, default="partialCostFunction",
                        help="Partial cost function folder name")
    parser.add_argument("--sourceType", dest="sourceType",
                        type=str,
                        help="Source type")
    parser.add_argument("--sourceFreq", dest="sourceFreq",
                        type=float,
                        help="Source frequency")
    parser.add_argument("--freqFilterList", dest="freqFilterList",
                        type=str, default="all",
                        help="Frequency filter list. \
                              Default is 'all'. \
                              Format: fq1,fq2,fq3")
    parser.add_argument("--gradDir", dest="gradDir",
                        type=str, default="partialGradient",
                        help="Directory to store the gradients")
    parser.add_argument("--minDepth", dest="minDepth",
                        type=float, default=0,
                        help="Minimal depth")
    parser.add_argument("--model", dest="model",
                        type=str, choices=["c", "1/c", "1/c2"],
                        help="Model considered")
    parser.add_argument("--method", dest="method",
                        type=str, default="steepestDescent",
                        help="Optimization method")
    parser.add_argument("--minimizerOptions", dest="miniOptions",
                        type=str, default="{}",
                        help="Options for the FWI minimizer")
    parser.add_argument("--taperOptions", dest="taperOptions",
                        type=str, default="{}",
                        help="Options for the sponge taper. Remember to \
                              set useTaper to True ")

    args, _ = parser.parse_known_args(hdrList)

    if args.NewParfile:
             # Check receiver position
        if (args.xmin>args.xmax) or (args.ymin>args.ymax):
            sys.exit("Error: x and y min must be smaller or equal to x and y max, respectively.")

        if args.nxr<1 or args.nyr<1:
            sys.exit("Error: nxr and nyr must be integers equal or greater than 1")

        if (args.xmin>=args.dsize) or (args.xmin<=0) or (args.xmax>=args.dsize) or (args.xmax<=0) or \
        (args.ymin>=args.dsize) or (args.ymin<=0) or (args.ymax>=args.dsize) or (args.ymax<=0):
            sys.exit("Error: xmin, xmax, ymin, and ymax must be less than dsize and bigger than 0.")

        if (args.xmin==args.xmax):
            args.xmax=args.xmax+0.1
            if args.nxr!=1:
                args.nxr=1

        if (args.ymin==args.ymax):
            args.ymax=args.ymax+0.1
            if args.nyr!=1:
                args.nyr=1

        # Check source position
        if (args.xsmin>args.xsmax) or (args.ysmin>args.ysmax):
            sys.exit("Error: xs and ys min must be smaller or equal to xs and ys max, respectively.")

        if args.Nxs<1 or args.Nys<1:
            sys.exit("Error: Nxs and Nys must be integers equal or greater than 1")

        if (args.xsmin>=args.dsize) or (args.xsmin<=0) or (args.xsmax>=args.dsize) or (args.xmax<=0) or \
        (args.ysmin>=args.dsize) or (args.ysmin<=0) or (args.ysmax>=args.dsize) or (args.ymax<=0):
            sys.exit("Error: xsmin, xsmax, ysmin, and ysmax must be less than dsize and bigger than 0.")

        if (args.xsmin==args.xsmax):
            args.xsmax=args.xsmax+0.1
            if args.Nxs!=1:
                args.Nxs=1

        if (args.ysmin==args.ysmax):
            args.ysmax=args.ysmax+0.1
            if args.Nys!=1:
                args.Nys=1

    return args

def get_model_parameters(args,key="model",comm=None):
    """
    Reads the velocity model parameters from an HDF5 file or generates a homogeneous model if no file is provided.

    Parameters:
    args (argparse.Namespace): Parsed arguments containing model parameters.
        - args.Pmodel (str, optional): Path to the HDF5 file containing the velocity model. If None, a homogeneous model is generated.
        - args.Ix, args.Iy (int): Number of elements along x and y directions (used if no model is provided).
        - args.vel (float): Default velocity value for the homogeneous model (used if no model is provided).
        - args.model (str, optional): Specifies the model type ("c", "1/c", or "1/c2") to adjust the velocity values.
        - key (str): Key to access the velocity data in the HDF5 file. Default is "model".
        - comm (mpi4py.MPI.Comm, optional): MPI communicator for parallel I/O. Default is None.

    Returns:
    tuple:
        - vel (numpy.ndarray): Velocity model as a 1D array. Adjusted based on the specified model type.
        - Ix, Iy (int): Number of elements along dimensions of the model.
    """
    if args.Pmodel is not None:
        velocity_file = args.Pmodel
        vf = h5py.File(velocity_file, 'r', driver="mpio", comm=comm)
        vel = vf[key][:].flatten()  # Ensure the model is always 1D
        vf.close()
        if args.Ix is None or args.Iy is None:
            Ix = Iy = int(np.sqrt(vel.shape[0]))
        else:
            Ix = args.Ix
            Iy = args.Iy

        if Ix * Iy != vel.shape[0]:
            raise ValueError(f"Model size mismatch: expected {Ix * Iy} elements, got {vel.shape[0]} elements in {velocity_file}. Parse correct value!")

    else:
        # define homogeneous model if no model was given
        if args.Ix is None or args.Iy is None:
            Ix = Iy = args.nbelem
        else:
            Ix = args.Ix
            Iy = args.Iy
        vel = np.zeros(Ix*Iy)
        vel[:] = args.vel

    # Allows solving for the parameters: 1/c or 1/c2 (acoustic only)
    if args.model == "1/c2":
        vel[vel == 0] = 1
        vel = 1/(vel*vel)
    elif args.model == "1/c":
        vel[vel == 0] = 1
        vel = 1/vel

    return vel,Ix,Iy
