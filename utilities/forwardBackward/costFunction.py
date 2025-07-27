import os

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np

import segyio

from ..imaging import globals as gb


class ComputeCostFunction:
    """
    Class containing methods associated with the computation of the cost function
    """
    def __init__( self, directory, dt, rootname ):
        """
        Parameters
        ------------
            directory : str
                Saving directory
            dt : float
                Time step in s
            rootname : str
                Rootname for the saved cost function file
        """
        self.directory = directory
        self.dt = dt
        self.rootname = rootname


    def computePartialCostFunction( self, residual ):
        """
        Computes the cost function for one shot given a residual

        Parameters
        ------------
            residual : array of float
                Array containing the residual

        """
        os.makedirs( self.directory, exist_ok=True )

        partialCost = 0
        for i in range(residual.shape[1]):
            partialCost += np.linalg.norm(residual[:, i])**2

        partialCost = self.dt * partialCost / 2

        with open(os.path.join(self.directory, self.rootname+".txt"), 'w') as f:
            f.write(str(partialCost))   


class ComputeResidual:
    """
    Contains methods used to compute the residual 
    """
    def __init__( self, dataFilename, filtering=None, interpolate=None ):
        """
        Parameters
        -----------
            dataFilename : str
                Filenames of observation data
            filtering : list
                List containing [callback, args] to filter the observation data
                Default is None (no filtering)
            interpolate : list
                List containing [callback, args] to interpolate the residual
                Default is None (no interpolation)
        """
        self.observationData = dataFilename
        self.filter = filtering
        self.interpolate = interpolate


    def __readObservationData( self ):
        """
        Extracts the observation/reference data from a SEGY file

        Parameters
        -----------
            filename : str
                Filename
        """
        with segyio.open( self.observationData, 'r', ignore_geometry=True) as f:
            data = np.zeros( (len(f.trace[0]), len(f.trace)) )
            for i in range( data.shape[1] ):
                data[:,i] = f.trace[i]

        self.data = data


    def computeResidual( self, simulatedData):
        """Computes the residual between the simulated and the observed data.

        Parameters
        ----------
            simulatedData : array
            Array containing the simulated data.

        Returns
        --------
            residual : array
            Array containing the residual, calculated as the difference 
            between the observed and the simulated data.
        """
        if not hasattr( self, "data"):
            self.__readObservationData()

        data = self.data

        # Apply filter if requested
        if self.filter is not None:
            filtering, *fargs = self.filter
            data = filtering( data, *fargs )
      
        # Interpolate if requested
        if self.interpolate is not None:
            interpolate, *args = self.interpolate
            data = interpolate( data, *args )
            
        # Compute residual (maximum time should be the same for both data)
        residual = data - simulatedData

        return residual


def computeFullCostFunction(nfiles, pCostDir="partialCostFunction", save=True, sfile="fullCostFunction.txt"):
    """Computes the full cost function by summing all partial cost functions

    Parameters
    ----------
        nfiles : int
            Total number of files/shots
        pCostDir : str, optional
            Directory containing the partial cost function files \
            Default is `partialCostFunction`
        save : bool, optional
            Save the full cost function value in a file \
            Default is True
        sfile : str, optional
            If save if True, filename of full cost function \
            Default is `fullCostFunction.txt`

    Returns
    --------
        fullCost : float
            Sum of all partial cost functions values
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        directory_in_str = pCostDir
        directory = os.fsencode(directory_in_str)

        fullCost = 0
        read = np.zeros((nfiles), dtype=int)
        while not all(read):
            if os.path.exists(directory_in_str):
                file_list = os.listdir(directory)

                if len(file_list) != 0:
                    for filename in file_list:
                        filename = os.fsdecode(filename)
                        shotId = int(os.path.splitext(filename.split("_")[-1])[0])
                        if os.stat(os.path.join(directory_in_str, filename)).st_size > 0 and not read[shotId-1]:
                            with open(os.path.join(directory_in_str, filename), "r") as costFile:
                                partialCost = float(costFile.readline())

                                fullCost += partialCost
                                read[shotId-1] = 1

                                if "DBG" not in gb.mode:
                                    os.remove(os.path.join(directory_in_str, filename))
                else:
                    continue

        if "DBG" not in gb.mode:
            os.rmdir(directory_in_str)
    else:
        fullCost = None
        
    fullCost = comm.bcast(fullCost, root=0)

    if save is True:
        writeCostFunction(fullCost, filename=sfile)

    return fullCost


def writeCostFunction(J, filename="fullCostFunction.txt"):
    """Write the cost function on disk
    
    Parameters
    -----------
        J : float
            CostFunction value
        filename : str, optional
            Name of the file to write on
    """
    with open(filename, 'a') as f:
        f.write(f"{str(J)}\n")
