import os
import shutil
#import numba
import numpy as np
import h5py
from scipy.fftpack import fftfreq, ifft, fft
import scipy.sparse as sp
from copy import deepcopy

from ..tools.domain import Domain, Subdomain
from ..tools.mpiComm import MpiDecomposition

from .SEMKernel import buildMassAcousticSEM, buildStiffnessAcousticSEM, buildMassTermSEM, \
                       buildDampingSEM, lagrange_nodal_weights, computeGradient

from .AcousticSolver2D import AcousticSolver2D
import time

class AcousticSolver2DRom(AcousticSolver2D):
    """
    Acoustic Spectral Elements (SEM) 2D solver.

    Based on the work of: Julien Besset - 2024

    It contains all methods to run a 2D AcousticSEM simulation using a python solver.

    Attributes
    -----------
        dt : float
            Time step for simulation
        minTime : float
            Minimum time considered in the simulation
        maxTime : float
            Maximum time considered in the simulation
        dtSeismo : float
            Time step to save pressure for seismic trace
        dtWaveField : float
            Time step to save fields
        sourceType : str
            Type of source
        sourceFreq : float
            Frequency of the source
        Sname : str
            Name of solver
            Fixed to SEM2D
        Stype : str
            Type of process using the solver
            Default is `modelling`
            Options are `modelling`/`modeling` or `inversion`/`fwi`
        L : list or float
            Size of the domain in meters
        Nxy : list or int
            Number of elements in x and y directions
        K : int
            Polynomial order of the elements
        bound : str
            Boundary conditions
        firstInit : bool
            Flag for initialize or reinitialize
        Isinit : bool
            Flag indicating if the solver is initialized
        minTimeSim : float
            Adjusted minimum simulation time
        maxTimeSim : float
            Adjusted maximum simulation time
        mpi : MpiDecomposition
            MPI decomposition object for parallel processing
        subdomain : Subdomain
            Subdomain object representing the local computational domain
        M : csc_matrix
            Mass matrix
        Mterm : csc_matrix
            Mass term matrix (used in inversion or FWI)
        R : csc_matrix
            Stiffness matrix
        S : csc_matrix
            Damping matrix
        sourceValue : ndarray
            Source values for the simulation
        PAtRecvs : ndarray
            Pressure values at receiver locations (seismogram)
        Pn0 : ndarray
            Wavefield at the previous time step
        Pn1 : ndarray
            Wavefield at the current time step
        Pn2 : ndarray
            Wavefield at the next time step
        Padj : ndarray
            Adjoint wavefield (used in inversion or FWI)
        grad : ndarray
            Gradient (used in inversion or FWI)
        modelparametrization : str
            Model parametrization type ("c", "1/c", or "1/c^2")
    """

    def __init__(self,
                 dt=None,
                 minTime=0.0,
                 maxTime=None,
                 dtSeismo=None,
                 dtWaveField=None,
                 sourceType=None,
                 sourceFreq=None,
                 Sname="SEM2DROM",
                 Stype="modelling",
                 L=None,
                 Nxy=None,
                 K=None,
                 bound="neumann",
                 freesurface=False,
                 orderFrechet=0,
                 orderGS=-1,
                 epsilonGS=0.01,
                 bufferSizeGS=100,
                 **kwargs):
        """
        Parameters
        ----------
            dt : float
                Time step for simulation
            minTime : float
                Starting time of simulation
                Default is 0
            maxTime : float
                End Time of simulation
            dtSeismo : float
                Time step to save pressure for seismic trace
            dtWaveField : float
                Time step to save fields
            sourceType : str
                Type of source
                Default is None
            sourceFreq : float
                Frequency of the source
                Default is None
            Sname : str
                Name of solver
                Fixed to SEM2D
            Stype : str
                Type of process using the solver
                Default is `modelling`
            L : 2-element list of floats or a float
                Size (in meters) of the domain
                Default is None
            Nxy : 2-element list of ints or an int
                Number of elements in x and y directions
                Default is None
            K : int
                Polynomial order of the elements
                Default is None
            bound : str
                Boundary conditions
                Default is neumann
            freesurface : bool
                Whether or not to use free surface boundary conditions on upper boundary
                Default is False
            **kwargs : dict
                Additional keyword arguments
                Default is None
        """

        super().__init__(dt,
                         minTime,
                         maxTime,
                         dtSeismo,
                         dtWaveField,
                         sourceType,
                         sourceFreq,
                         Sname,
                         Stype,
                         L,
                         Nxy,
                         K,
                         bound,
                         freesurface,
                         **kwargs)

        self.orderFrechet = orderFrechet
        self.orderGS = orderGS
        self.epsilonGS = epsilonGS
        self.bufferSize = bufferSizeGS
        self.alpha = 0
        self.solverROM = False


    def __repr__(self):
        string_list = []
        string_list.append("Solver name : " + self.Sname + "\n")
        string_list.append("Process using solver : " + self.Stype + "\n")
        string_list.append("dt : " + str(self.dt) + "\n")
        string_list.append("maxTime : " + str(self.maxTime) + "\n")
        string_list.append("dtSeismo : " + str(self.dtSeismo) + "\n")
        rep = ""
        for string in string_list:
            rep += string

        return rep



    def setReducedModel(self, comm):

        self.readReducedMatrices(comm)
        self.Mrom += self.alpha * self.Mprom
        self.Srom += self.alpha * self.Sprom
        MpSrom = self.Mrom + 0.5 * self.dt * self.Srom
        self.iMpSrom = np.linalg.inv(MpSrom)

        self.an0 = np.zeros(self.Mrom.shape[0])
        self.an1 = np.zeros(self.Mrom.shape[0])
        self.an2 = np.zeros(self.Mrom.shape[0])

        self.From = np.zeros(self.Mrom.shape[0])


    def execute(self, timesample=0, dirflag="forward", comm=None):
        """
        Executes a single time step of the 2D acoustic solver.

        This method performs the following steps:
        1. Constructs the source term vector for the current time step.
        2. Updates the past wavefield values to prepare for the next time step.
        3. Solves the linear system to compute the wavefield at the next time step.
        4. Updates the wavefield with the computed solution.
        5. Handles forward or backward propagation based on the `dirflag` parameter.

        Parameters:
        -----------
        timesample : int
            The current time step index for the simulation.
        dirflag : str
            Direction of propagation. Must be either "forward" or "backward".
        """
        # Check the size of the source value

        maxCycle = int(round((self.maxTime-self.minTimeSim)/self.dt))
        if not self.solverROM:
            super().execute(timesample, dirflag, comm)
            timesample -= int(np.round(self.minTimeSim/self.dt))

            if (timesample%20==0 or timesample+2 == maxCycle) and self.orderGS >= 0:
                self.bufferROM[0, self.bufferIter[0], :] = self.Pn2[:]
                self.bufferTimeIndex[0, self.bufferIter[0]] = timesample + 2
                self.bufferIter[0] +=1
                if (self.bufferIter[0]>0 and self.bufferIter[0]%self.bufferSize==0) or timesample+2 == maxCycle:
                    self.gramSchmidtMPIbuffer(0, comm)
                    self.bufferIter[0] = 0
                    self.bufferROM[0,:,:] = 0.0


            if "forward" in dirflag.lower() and self.orderFrechet>0:
                dt2 = self.dt**2
                irom = int(round(1/self.maxfreq/8/self.dt))
                for f in range(self.orderFrechet):
                    if f==0:
                        rhs = (2*self.M - dt2*self.R)*self.Pfn1[f,:] - (self.M - 0.5*self.dt*self.S)*self.Pfn0[f,:] - (f+1)*self.Mp * (self.Pn2 - 2*self.Pn1 + self.Pn0)
                    else:
                        rhs = (2*self.M - dt2*self.R)*self.Pfn1[f,:] - (self.M - 0.5*self.dt*self.S)*self.Pfn0[f,:] - (f+1)*self.Mp * (self.Pfn2[f-1,:] - 2*self.Pfn1[f-1,:] + self.Pfn0[f-1,:])

                    self.Pfn2[f,:] = self.solve(rhs)
                    if self.freesurface:
                        # Apply free surface condition if needed
                        self.Pfn2[f,self.subdomain.upperNodes] = 0.0

                    self.Pfn2[f,:] = self.mpi.synchronize(self.Pfn2[f,:], self.subdomain)
                    if (timesample%self.irom==0 or timesample+2 == maxCycle) and self.orderGS >= f+1:
                        self.bufferROM[f+1, self.bufferIter[f+1], :] = self.Pfn2[f,:]
                        self.bufferTimeIndex[f+1, self.bufferIter[f+1]] = timesample + 2
                        self.bufferIter[f+1] += 1
                        if (self.bufferIter[f+1]>0 and self.bufferIter[f+1]%self.bufferSize==0) or timesample+2 == maxCycle:
                            self.gramSchmidtMPIbuffer(f+1, comm)
                            self.bufferIter[f+1] = 0
                            self.bufferROM[f+1,:,:] = 0.0

                if timesample+2 == maxCycle and self.orderGS >= 0:
                    self.createFinalBasis(comm)
                    self.writeReducedMatrices(comm)

        else:
            dt2 = self.dt**2
            timesample -= int(np.round(self.minTimeSim/self.dt)) # Make sure timesample starts at 0
            self.updateForwardSourceTerm(self.sourceValue[timesample,:])
            self.updatePastWaveField()

            rhs =  2*np.dot(self.Mrom, self.an1) + dt2 * (self.From - self.an1) - np.dot(self.Mrom - 0.5*self.dt*self.Srom, self.an0)
            self.an2 = np.dot(self.iMpSrom, rhs)

    def setDirectionFrechet(self, filename, comm):
        """
        Update the direction of perturbation for Frechet derivate computation

        Parameters
        ----------
            perturbation : array/list
                List/array containing value per cell of the direction of perturbation
        """
        if filename is not None:
            while True:
                try:
                    h5f = h5py.File(filename, driver="mpio", comm=comm)
                    key = list(h5f.keys())[0]
                    grad = h5f[key][:]
                    h5f.close()
                    self.perturbation = grad[self.subdomain.globElems]/max(abs(grad))
                    break
                except:
                    time.sleep(0.5)

            model1 = self.perturbation.reshape((self.subdomain.Iy,self.subdomain.Ix)).T
            model2 = self.velocity.reshape((self.subdomain.Iy,self.subdomain.Ix)).T
            self.Mp = buildMassAcousticSEM(self.subdomain, model1)
            self.Sp = buildDampingSEM(self.subdomain, 0.5 * model1[:,:] * model2[:,:])

        if self.orderFrechet > 0 and filename is not None:
            self.Pfn0 = np.zeros((self.orderFrechet, self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
            self.Pfn1 = np.zeros((self.orderFrechet, self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
            self.Pfn2 = np.zeros((self.orderFrechet, self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        elif self.orderFrechet > 0 and filename	is None:
            raise ValueError("You must provide a direction to compute the Frechet derivative.")

        self.sizePhi = 50
        self.phi = np.zeros((0, self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        self.sizeROMFrechet = np.zeros(self.orderFrechet+1, dtype=np.int64)
        self.bufferROM = np.zeros((self.orderFrechet+1, self.bufferSize, self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        self.bufferTimeIndex = np.zeros((self.orderFrechet+1, self.bufferSize), dtype=np.int64)
        self.bufferIter = np.zeros(self.orderFrechet+1, dtype=np.int64)
        self.phiFrechet = np.zeros((self.orderFrechet+1, self.sizePhi , self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        self.snapshotsTime = np.zeros((self.orderFrechet+1, self.sizePhi), dtype=np.int64)
        self.irom = int(round(1/self.maxfreq/10/self.dt))

    def computeSrcAndRcvConstants(self, sourcesCoords=[], receiversCoords=[], comm=None):

        # If lists are empty source and receivers constants are not updated.
        if not self.solverROM:
            self.precomputeRcv(receiversCoords)
            self.precomputeSrc(sourcesCoords)
        else:
            self.readReducedSrcandRcv(comm)


    def updateForwardSourceTerm(self, sourceval):
        """
        Constructs the source term vector by projecting the actual source value(s)
        onto the computational domain using Lagrange basis functions.

        Args:
            sourceval (float or nump.ndarray): The value(s) of the source to be applied at the nodes
            associated with the source position(s).

        """
        if not self.solverROM:
            super().updateForwardSourceTerm(sourceval)
        else:
            self.From[:] = 0.0
            sourceval = np.asarray(sourceval)

            for source_index in range(len(self.SrcNodes)):
                self.From[:] += sourceval[source_index] * self.SrcNodalWeightsROM[source_index]


    def resetWaveFields(self, resetall=False):
        """
        Resets the wavefields used in the simulation.

        This method reinitializes the arrays representing the wavefields at
        different time steps (Pn0, Pn1, Pn2) to zero. If `resetall` is True,
        it also resets the adjoint wavefield (Padj) and the gradient (grad) to zero.
        """

        if not self.solverROM:

            self.Pn0[:] = 0.0
            self.Pn1[:] = 0.0
            self.Pn2[:] = 0.0

            if self.orderFrechet>0:
                self.Pfn0[:,:] = 0.0
                self.Pfn1[:,:] = 0.0
                self.Pfn2[:,:] = 0.0

            if self.Stype.lower() in ("inversion","fwi") and resetall:
                # Reset Adjoint wavefield
                self.Padj[:] = 0.0
                # Reset Gradient
                self.grad[:] = 0.0

        else:
            self.an0[:] = 0.0
            self.an1[:] = 0.0
            self.an2[:] = 0.0


    def updatePastWaveField(self):
        """
        Updates the past wavefield values for the simulation.

        This method shifts the wavefield arrays to prepare for the next time step
        in the simulation. Specifically, it moves the current wavefield (Pn1) to
        the past wavefield (Pn0) and the future wavefield (Pn2) to the current
        wavefield (Pn1).

        Attributes:
        -----------
        Pn0 : numpy.ndarray
            The wavefield at the previous time step.
        Pn1 : numpy.ndarray
            The wavefield at the current time step.
        Pn2 : numpy.ndarray
            The wavefield at the next time step.
        """

        if not self.solverROM:
            self.Pn0[:] = self.Pn1[:]
            self.Pn1[:] = self.Pn2[:]

            if self.orderFrechet>0:
                self.Pfn0[:,:] = self.Pfn1[:,:]
                self.Pfn1[:,:] = self.Pfn2[:,:]

        else:
            self.an0[:] = self.an1[:]
            self.an1[:] = self.an2[:]



    def updatePressureAtReceivers(self, timesample=0, flag="forward"):
        """
        Updates the pressure values at receiver locations for a given time sample.
        In other words, it calculates the seismogram samples at each time step.

        This method calculates the pressure at specified receiver coordinates
        using the current wavefield (Pn2) and updates the pressure array `PAtRecvs`
        (the seismogram) for the given time sample.

        Parameters:
        -----------
        timesample : int
            The time sample index at which the pressure is being updated.
        flag : str
            Direction of propagation. Must be either "forward" or "backward".
            If "forward", self.PAtRecvs will be updated based on the position of the receivers,
            otherwise the position of the source(s) will be considered.
        """

        if not self.solverROM:
            super().updatePressureAtReceivers(timesample, flag)
        else:
            for receiver_index in range(len(self.RcvNodes)):
                if self.RcvNodes[receiver_index] is not None:
                    self.PAtRecvs[timesample, receiver_index] = np.dot(self.an2, self.RcvNodalWeightsROM[receiver_index])


    def gramSchmidtMPIbuffer(self, order, comm):
        nx = self.phiFrechet.shape[2]

        w = np.zeros(nx)
        v = np.zeros(nx)
        q = np.zeros(nx)
        nodes = self.subdomain.nodesList

        for i in range(self.bufferSize):

            v[:] = self.bufferROM[order,i,:]
            w[:] = self.R.dot(v)

            n0t = np.dot(v[nodes], w[nodes])
            n0 = comm.allreduce(n0t)
            n = n0
            success = 1
            for j in range(self.sizeROMFrechet[order]-1,-1,-1):
                q[:] = self.phiFrechet[order,j,:]
                spqt = np.dot(q[nodes], w[nodes])
                spq = comm.allreduce(spqt)
                n -= spq**2
                if n < self.epsilonGS*n0:
                    success = 0
                    break
                else:
                    v -= spq * q

            if success and n > self.epsilonGS*n0:
                self.sizeROMFrechet[order] += 1
                if self.sizeROMFrechet[order] >= self.sizePhi:
                    old_size = self.sizePhi
                    self.sizePhi *= 2
                    new_phi = np.zeros((self.orderFrechet + 1, self.sizePhi, nx))
                    new_phi[:, :old_size, :] = self.phiFrechet
                    self.phiFrechet = new_phi

                    new_snapshots = np.zeros((self.orderFrechet + 1, self.sizePhi))
                    new_snapshots[:, :old_size] = self.snapshotsTime
                    self.snapshotsTime = new_snapshots

                temp = self.R.dot(v)
                spvt = np.dot(v[nodes], temp[nodes])
                spv = comm.allreduce(spvt)
                self.phiFrechet[order, self.sizeROMFrechet[order]-1, :] = v/np.sqrt(spv)
                self.snapshotsTime[order, self.sizeROMFrechet[order]-1] = self.bufferTimeIndex[order,i]

                if self.sizeROMFrechet[order]%10 == 0:
                    self.reorthogonalizationMPI(order, comm)


    def reorthogonalizationMPI(self, order, comm):
        nx = self.phiFrechet.shape[2]
        w = np.zeros(nx)
        v = np.zeros(nx)
        q = np.zeros(nx)
        nodes = self.subdomain.nodesList

        ii = 0
        for i in range(self.sizeROMFrechet[order]):
            v[:] = self.phiFrechet[order,i,:]
            w[:] = self.R.dot(v)
            n0t = np.dot(v[nodes],w[nodes])
            n0 = comm.allreduce(n0t)
            n = n0
            success = 1

            for j in range(ii):
                q[:] = self.phiFrechet[order,j,:]
                spqt = np.dot(q[nodes],w[nodes])
                spq = comm.allreduce(spqt)
                n -= spq**2
                if n < n0*self.epsilonGS:
                    success = 0
                    break
                elif abs(spq) > 1e-8:
                    v -= spq * q

            if success and n > n0*self.epsilonGS:
                temp = self.R.dot(v)
                spvt = np.dot(v[nodes],temp[nodes])
                spv = comm.allreduce(spvt)
                self.phiFrechet[order, ii, :] = v/np.sqrt(spv)
                self.snapshotsTime[order,ii] = self.snapshotsTime[order,i]
                ii+=1

        self.phiFrechet[order, ii:,:] = 0.0
        self.sizeROMFrechet[order] = ii
        self.snapshotsTime[order, ii:] = 0


    def modifiedGramSchmidtMPI(self, A, comm):
        nx = A.shape[1]

        rank = comm.Get_rank()

        w = np.zeros(nx)
        v = np.zeros(nx)
        q = np.zeros(nx)
        nodes = self.subdomain.nodesList
        sizeROM = 0

        for k in range(A.shape[0]):

            v[:] = A[k,:]
            w[:] = self.R.dot(v)
            n0t = np.dot(v[nodes], w[nodes])
            n0 = comm.allreduce(n0t)

            if n0 < self.epsilonGS:
                continue

            sizeROM+=1
            r = np.sqrt(n0)
            q[:] = v[:]/r
            self.phi = np.resize(self.phi, (sizeROM, nx))
            self.phi[sizeROM-1,:] = q[:]

            w[:] /= r
            for i in range(k+1, A.shape[0]):
                spt = np.dot(w[nodes], A[i,nodes])
                sp = comm.allreduce(spt)
                A[i,:] -= sp * q[:]



    def createFinalBasis(self, comm):
        snapshots = np.concatenate([self.phiFrechet[f, :self.sizeROMFrechet[f], :] for f in range(self.orderFrechet+1)], axis=0)
        self.phiFrechet[:,:,:] = 0.0

        snapshotsTime = np.concatenate([self.snapshotsTime[f, :self.sizeROMFrechet[f]] for f in range(self.orderFrechet+1)])
        ind = np.argsort(snapshotsTime)
        snapshots = snapshots[ind,:]

        self.modifiedGramSchmidtMPI(snapshots, comm)

        if comm.Get_rank() == 0:
            print("\nNumber of ROM basis functions:", self.phi.shape[0])




    def writeReducedMatrices(self, comm):
        directory = "phi/shot_"+str(self.shotId)+"/rank_"+str(comm.Get_rank())
        os.makedirs(directory, exist_ok=True)

        nodes = self.subdomain.nodesList
        phi = self.phi[:,nodes]
        M = self.M[np.ix_(nodes,nodes)]
        Mp = self.Mp[np.ix_(nodes,nodes)]
        S = self.S[np.ix_(nodes,nodes)]
        Sp = self.Sp[np.ix_(nodes,nodes)]

        Mrom = phi @ M @ phi.T

        h5f = h5py.File(directory+"/M.hdf5",'w')
        h5f.create_dataset("value", data=Mrom)
        h5f.close()

        Srom = phi @ S @ phi.T

        h5f = h5py.File(directory+"/S.hdf5",'w')
        h5f.create_dataset("value", data=Srom)
        h5f.close()

        Mprom = phi @ Mp @ phi.T

        h5f = h5py.File(directory+"/Mp.hdf5",'w')
        h5f.create_dataset("value", data=Mprom)
        h5f.close()

        Sprom = phi @ Sp @ phi.T

        h5f = h5py.File(directory+"/Sp.hdf5",'w')
        h5f.create_dataset("value", data=Sprom)
        h5f.close()


        Srcrom = np.zeros((self.SrcNodes.shape[0], self.phi.shape[0]))
        for isrc in range(self.SrcNodes.shape[0]):
            Srcrom[isrc,:] = self.phi[:,self.SrcNodes[isrc]] @ self.SrcNodalWeights[isrc]

        h5f = h5py.File(directory+"/Src.hdf5",'w')
        h5f.create_dataset("value", data=Srcrom)
        h5f.close()

        Rcvrom = np.zeros((self.RcvNodes.shape[0], self.phi.shape[0]))
        for ircv in range(self.RcvNodes.shape[0]):
            Rcvrom[ircv,:] = self.phi[:,self.RcvNodes[ircv]] @ self.RcvNodalWeights[ircv]

        h5f = h5py.File(directory+"/Rcv.hdf5",'w')
        h5f.create_dataset("value", data=Rcvrom)
        h5f.close()



    def readReducedMatrices(self, comm):

        directory = "phi/shot_"+str(self.shotId)+"/rank_"+str(comm.Get_rank())

        h5f = h5py.File(directory+"/M.hdf5",'r')
        Mrom = h5f["value"][:,:]
        h5f.close()

        self.Mrom = comm.allreduce(Mrom)

        h5f = h5py.File(directory+"/Mp.hdf5",'r')
        Mprom = h5f["value"][:,:]
        h5f.close()

        self.Mprom = comm.allreduce(Mprom)

        h5f = h5py.File(directory+"/S.hdf5",'r')
        Srom = h5f["value"][:,:]
        h5f.close()

        self.Srom = comm.allreduce(Srom)

        h5f = h5py.File(directory+"/Sp.hdf5",'r')
        Sprom = h5f["value"][:,:]
        h5f.close()

        self.Sprom = comm.allreduce(Sprom)


    def readReducedSrcandRcv(self, comm):

        directory = "phi/shot_"+str(self.shotId)+"/rank_"+str(comm.Get_rank())

        h5f = h5py.File(directory+"/Src.hdf5",'r')
        SrcNodalWeightsROM = h5f["value"][:,:]
        h5f.close()

        self.SrcNodalWeightsROM = comm.allreduce(SrcNodalWeightsROM)

        h5f = h5py.File(directory+"/Rcv.hdf5",'r')
        self.RcvNodalWeightsROM = h5f["value"][:,:]
        h5f.close()
