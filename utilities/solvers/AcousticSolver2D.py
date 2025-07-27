import os
import shutil

import numpy as np
import h5py
from scipy.fftpack import fftfreq, ifft, fft
from scipy.sparse.linalg import factorized

from ..tools.domain import Domain, Subdomain
from ..tools.mpiComm import MpiDecomposition

from .SEMKernel import buildMassAcousticSEM, buildStiffnessAcousticSEM, buildMassTermSEM, \
                       buildDampingSEM, lagrange_nodal_weights, computeGradient


class AcousticSolver2D:
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
                 Sname="SEM2D",
                 Stype="modelling",
                 L=None,
                 Nxy=None,
                 K=None,
                 bound="neumann",
                 freesurface=False,
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
        assert isinstance(dt, (float, type(None))), "dt must be a float or None"
        assert isinstance(minTime, (float, type(None))), "minTime must be a float or None"
        assert isinstance(maxTime, (float, type(None))), "maxTime must be a float or None"
        assert isinstance(dtSeismo, (float, type(None))), "dtSeismo must be a float or None"
        assert isinstance(dtWaveField, (float, type(None))), "dtWaveField must be a float or None"
        assert isinstance(sourceType, (str, type(None))), "sourceType must be a string or None"
        assert isinstance(sourceFreq, (float, type(None))), "sourceFreq must be a float or None"
        assert isinstance(Stype, str), "Stype must be a string"
        assert isinstance(L, (list, float, type(None))), "L must be a list, float, or None"
        assert isinstance(Nxy, (list, int, type(None))), "Nxy must be a list, int, or None"
        assert isinstance(K, (int, type(None))), "K must be an int or None"
        assert isinstance(bound, str), "bound must be a string"
        assert isinstance(freesurface, bool), "freesurface must be a boolean"

        if dt is not None:
            assert dt > 0, "dt must be bigger than 0"
        self.dt = dt
        self.minTime = minTime
        self.minTimeSim = minTime
        self.maxTime = maxTime
        self.maxTimeSim = maxTime
        # dtSeismo must be a multiple of dt
        if dtSeismo is not None and dt is not None:
            if dtSeismo > dt:
                ratio = dtSeismo / dt
                if not ratio.is_integer():
                    if ratio > 1:
                        dtSeismo = round(ratio) * dt
                    else:
                        dtSeismo = dt
            elif dtSeismo < dt:
                dtSeismo = dt
        self.dtSeismo = dtSeismo
        # dtWaveField must be bigger or equal to dtSeismo and also a multiple of dtSeismo
        if dtWaveField is not None and dtSeismo is not None:
            if dtWaveField > dtSeismo:
                ratio = dtWaveField / dtSeismo
                if not ratio.is_integer():
                    if ratio > 1:
                        dtWaveField = round(ratio) * dtSeismo
                    else:
                        dtWaveField = dtSeismo
            elif dtWaveField < dtSeismo:
                dtWaveField = dtSeismo
        self.dtWaveField = dtWaveField
        self.sourceType = sourceType
        self.sourceFreq = sourceFreq
        assert Stype.lower() in ("modelling", "modeling", "inversion", "fwi"), \
            "Stype takes the values: 'modelling', 'modeling', 'inversion', or 'fwi'."
        self.Stype = Stype
        self.Sname = Sname
        self.L = L
        self.Nxy = Nxy
        self.K = K
        self.bound = bound
        self.freesurface = freesurface
        self.firstInit = True
        self.Isinit = False
        self.minTimeSim = minTime
        self.maxTimeSim = maxTime
        self.args = kwargs


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


    def initialize(self, rank=0, comm=None, vel=None):
        """
        Initializes the 2D Spectral Element Method (SEM) solver.

        This method sets up the computational domain, MPI decomposition, and subdomain
        for the solver. It also applies the initial conditions to the system.

        rank : int, optional
            The process rank in a parallel computation. Default is 0.
        comm : MPI communicator, optional
            The MPI communicator object for parallel execution. Default is None.
        vel : ndarray, optional
            The velocity model used for the simulation. Default is None.

        Attributes Set
        ---------------
        Isinit : bool
            A flag indicating whether the solver has been initialized.
        firstInit : bool
            A flag used to track the initialization state during the setup process.
        subdomain : Subdomain
            The subdomain object representing the local computational domain.
        mpi : MpiDecomposition
            The MPI decomposition object for parallel processing.

        Example
        -------
        >>> solver = AcousticSolver2D()
        >>> solver.initialize(rank=0, comm=MPI.COMM_WORLD, vel=velocity_model)

        """

        if rank==0:
            print("Initializing solver")

        if not self.firstInit:
            raise RuntimeError("Solver can only be initialized once. Reinitialize instead.")

        if self.dt is None:
            raise ValueError("dt must be set before initializing the solver")

        if self.dtSeismo is None:
            self.dtSeismo = self.dt

        if self.dtWaveField is None:
            self.dtWaveField = self.dtSeismo

        if isinstance(self.L, list) and all(isinstance(i, float) for i in self.L):
            Lx = self.L[0]
            if len(self.L) > 1:
                Ly = self.L[1]
            else:
                Ly = self.L[0]
        else:
            Lx = self.L
            Ly = self.L

        if isinstance(self.Nxy, list) and all(isinstance(i,int) for i in self.Nxy):
            Ix = self.Nxy[0]
            if len(self.Nxy) > 1:
                Iy = self.Nxy[1]
            else:
                Iy = self.Nxy[0]
        else:
            Ix = self.Nxy
            Iy = self.Nxy

        if isinstance(self.K,int):
            K = self.K
        else:
            raise ValueError("K must be an int")

        domain = Domain(Lx=Lx,Ix=Ix,Ly=Ly,Iy=Iy,K=K,boundary_conditions=self.bound,freesurface=self.freesurface)
        self.mpi = MpiDecomposition(comm)
        self.subdomain = Subdomain(domain,self.mpi)

        self.velocity = vel[self.subdomain.globElems] if vel is not None else None

        self.Isinit = True
        self.precomputeSEMmatrices(comm)
        # Set taper
        useTaper = self.args.get("useTaper", False) # Must be set to True to activate taper
        inUpper = self.args.get("inUpper", False)
        exportTaper = self.args.get("exportTaper", False)
        spongeWidth = self.args.get("spongeWidth", 0.1)
        spongeFactor = self.args.get("spongeFactor", 1.0)
        # Array containing taper values
        self.Taper = np.ones((self.subdomain.nx*self.subdomain.ny), dtype=np.float64)
        # Compute sponge values (if needed)
        if useTaper:
            self.initspongeTaper(spongeWidth=spongeWidth, spongeFactor=spongeFactor, inUpper=inUpper, vMax=max(vel), exportTaper=exportTaper, comm=comm)

        # Initialize the source term vector
        self.F = np.zeros((self.subdomain.nx*self.subdomain.ny), dtype=np.float64)
        # Initialize source values
        self.getsrcvalues()


    def initFields(self,nxyr=1):
        """
        Initializes the fields required for the 2D acoustic solver.

        This method sets up the arrays used during a shot simulation: the P-wave seismogram,
        P wavefields, adjoint wavefield, and gradient.

        Args:
            nxyr (int): The number of receiver points (per shot) in the simulation.

        Attributes:
            PAtRecvs (numpy.ndarray): A 2D array of shape (Nt, nxyr) to store the
                P-wave seismogram at receiver locations, where Nt is the number of
                time steps.
            Pn1 (numpy.ndarray): A 2D array of shape (subdomain.nx * subdomain.ny, Nt)
                to store the P-wave wavefield at time t+1.
            Padj (numpy.ndarray): A 2D array of shape (subdomain.nx * subdomain.ny, Nt)
                to store the adjoint wavefield. Only initialized if the simulation
                type is "inversion" or "fwi".
            grad (numpy.ndarray): A 1D array of shape (subdomain.nx * subdomain.ny)
                to store the gradient. Only initialized if the simulation type is
                "inversion" or "fwi".
        """

        if self.dtSeismo != self.dt:
            Nt = (self.maxTime - self.minTime) / self.dtSeismo
        else:
            Nt = (self.maxTime - self.minTime) / self.dt
        Nt = int(round(Nt)) + 1

        # P-wave Seismogram
        self.PAtRecvs = np.zeros((Nt,nxyr),dtype=np.float64)
        # P-wave Wavefield at t-1, t, and t+1
        self.Pn0 = np.zeros((self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        self.Pn1 = np.zeros((self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
        self.Pn2 = np.zeros((self.subdomain.nx*self.subdomain.ny), dtype=np.float32)

        if self.Stype.lower() == "inversion" or self.Stype.lower() == "fwi":
            # Adjoint wavefield
            self.Padj = np.zeros((self.subdomain.nx*self.subdomain.ny), dtype=np.float32)
            # Gradient
            self.grad = np.zeros((self.subdomain.Ix*self.subdomain.Iy), dtype=np.float64)


    def reinitSolver(self, comm=None):
        """
        Reinitialize the 2D SEM acoustic solver.

        This method resets the solver's initialization state and recalculates
        the mass, stiffness, and damping matrices for the subdomain based on
        the velocity model. It prepares the solver for subsequent computations.

        Steps:
        -------
        1. Extract the velocity model for the subdomain using global element indices.
        2. Reshape the velocity model to match the subdomain's grid dimensions.
        3. Compute the mass and stiffness matrices using the Spectral Element Method (SEM).
        4. Compute the damping matrix using the SEM.
        5. Calculate the modified mass matrices (MpS and MmS) incorporating damping.

        Notes:
        ------
        - `subdomain` contains information about the local subdomain, including
          global element indices (`globElems`) and grid dimensions (`Ix`, `Iy`).
        - `dt` is the time step used in the solver.
        - This method should be called when the solver needs to be reinitialized
          after changes to the velocity model or other parameters.

        Attributes:
        -----------
        - self.firstInit (bool): A flag indicating if it is the first initialization
          of the solver. Set to False to indicate reinitialization.
        """
        if self.Isinit:
            self.precomputeSEMmatrices(comm)
        else:
            raise RuntimeError("Solver must be initialized before reinitialization.")

        if self.firstInit:
            self.firstInit = False


    def precomputeSEMmatrices(self, comm=None):
        """
        Compute SEM matrices right after (re)initialization.
        This method sets up the initial matrices for the solver by:
        - Retrieving the velocity model for the current subdomain.
        - Reshaping the velocity model to match the subdomain's grid dimensions.
        - Building the mass and stiffness matrices using the SEM (Spectral Element Method).
        - Constructing the damping matrix using the SEM.
        - Calculating the modified mass matrices (MpS and MmS) by incorporating the damping effects.
        Note:
            This function assumes that the solver's state is not yet initialized.
        """

        if self.Isinit:

            model = self.velocity.reshape((self.subdomain.Iy,self.subdomain.Ix)).T

            self.S = buildDampingSEM(self.subdomain, 1/model)
            self.M = buildMassAcousticSEM(self.subdomain, 1/(model**2))

            A = self.M + 0.5 * self.dt * self.S
            self.solve = factorized(A)

        else:
            print("Bad call to precomputeSEMmatrices? Initialize solver first")
            pass

        # Pre-factorize for faster time step computations
        A = self.M + 0.5 * self.dt * self.S
        
        self.solve = factorized(A)
        del A
        # Calculate mass matrix without model, the mass matrix term used when computing the gradient.
        # Also calculate stiffness matrix. In the acoustic case it does not depend on the model.
        if self.firstInit:

            self.R = buildStiffnessAcousticSEM(self.subdomain)

            if self.Stype.lower() == "inversion" or self.Stype.lower() == "fwi":
                self.Mterm = buildMassTermSEM(self.subdomain)


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
        if "forward" in dirflag.lower():
            timesample -= int(np.round(self.minTimeSim/self.dt)) # Make sure timesample starts at 0
            self.updateForwardSourceTerm(self.sourceValue[timesample,:])
        elif "backward" in dirflag.lower():
            self.updateBackwardSourceTerm(self.sourceValue[timesample,:])
        else:
            raise ValueError("dirflag must be 'forward' or 'backward'")

        self.updatePastWaveField()
        
        rhs = (2*self.M - self.dt**2*self.R)*self.Pn1 - (self.M - 0.5*self.dt*self.S)*self.Pn0 + self.dt**2 * self.F
        self.Pn2[:] = self.solve(rhs)
        if self.freesurface:
            # Apply free surface condition if needed
            self.Pn2[self.subdomain.upperNodes] = 0.0
        # Apply taper to the wavefield
        #self.Pn2[:] *= self.Taper[:]
        self.Pn2[:] = self.mpi.synchronize(self.Pn2, self.subdomain)


    def finalize(self):
        """
        Finalize simulation. Sets the solver to None and cleans up the memory.
        This should be called at the end of the simulation.
        """
        if self.Isinit:
            print("Finalizing solver")
            self.Stype = None
            self.firstInit = True
            self.Isinit = False

            # Reset source values and frequencies
            self.sourceValue = None
            self.sourceFreq = None
            self.sourceType = None
            self.dt = None
            self.minTime = None
            self.maxTime = None
            self.dtSeismo = None
            self.dtWaveField = None

            print("Solver finalized and resources cleaned up.")


    def outputVtk(self, time):
        """
        Output VTK with wavefields stored during propagation.

        Parameters
        ----------
            time : float
                Current time of simulation
        """
        # TODO
        pass


    def getsrcvalues(self):
        """
        Evaluate source used in the simulation.
        Only Ricker wavelets of order {0 - 4} are accepted for the moment.

        ricker0 corresponds to the standard Gaussian wavelet.
        ricker{1-4} corresponds to the {1-4}-order derivatives of the Gaussian wavelet.
        """
        sourceTypes = ("ricker0", "ricker1", "ricker2", "ricker3", "ricker4")
        assert self.sourceType in sourceTypes, f"Only {sourceTypes} are allowed"

        f0 = self.sourceFreq
        self.maxfreq = 2.5*f0
        #delay = 1.0 / f0 # Not efficient
        delay = 1.2 / f0 # Ricker "golden rule" as usually defined (best for safeguard)
        #delay = 2.0*np.sqrt(np.pi) / (3*f0) # Close to the value of the "golden rule", assumes a cut frequency of 3*f0
        alpha = - ( f0 * np.pi )**2

        nsamples = int( round( ( self.maxTime - self.minTime ) / self.dt )) + 1
        sourceValue = np.zeros(( nsamples, 1 ))

        order = int( self.sourceType[-1] )
        sgn = ( -1 )**( order + 1 )

        time = self.minTime
        for nt in range(nsamples):

            if self.minTime <= - delay:
                # Where does this empirical value comes from?
                # Maybe it should be 2x the delay?
                tmin = - 2.9 / f0
                tmax = 2.9 / f0
                time_d = time

            else:
                time_d = time - delay
                tmin = 0.0
                tmax = 2.9 / f0

            if (time > tmin and time < tmax) or ( self.minTime < - delay and time == tmin ):
                gaussian = np.exp( alpha * time_d**2)

                if order == 0:
                    sourceValue[nt, 0] = sgn * gaussian

                elif order == 1:
                    sourceValue[nt, 0] = sgn * ( 2 * alpha * time_d ) * gaussian

                elif order == 2:
                    sourceValue[nt, 0] = sgn * ( 2 * alpha + 4 * alpha**2 * time_d**2 ) * gaussian

                elif order == 3:
                    sourceValue[nt, 0] = sgn * ( 12 * alpha**2 * time_d + 8 * alpha**3 * time_d**3 ) * gaussian

                elif order == 4:
                    sourceValue[nt, 0] = sgn * ( 12 * alpha**2  + 48 * alpha**3 * time_d**2 + 16 * alpha**4 * time_d**4 ) * gaussian

            time += self.dt

        self.updateSourceFrequency(self.sourceFreq)
        self.updateSourceValue(sourceValue)
        #self.updateSourceValue(sourceValue/max(abs(sourceValue))) # Normalize the source value


    def updateSourceValue(self, value):
        """
        Update the source values in the simulation.

        Also sets the source term vector F to zero.

        Parameters
        ----------
            value : array/list
                List/array containing the value of the source for each time step
        """

        assert isinstance(value, (list, np.ndarray)), "value must be a list or numpy array"

        self.sourceValue = np.copy(value) if isinstance(value, np.ndarray) else value

        self.F[:] = 0


    def updateSourceFrequency(self, freq):
        """
        Overwrite source frequency

        Parameters
        ----------
            freq : float
                Frequency of the source in Hz
        """

        self.sourceFreq = freq


    def filterSource(self, fmax):
        """
        Filter the source function. Note that this will modify the starting time of the forward simulation to avoid discontinuity.

        Parameters
        -----------
            fmax : float/string
                Cut-off frequency (Hz). Frequencies above this value are attenuated.
                After filtering, the frequency band of the new source will be contained in the interval [0,fmax+1].
        """
        if str(fmax) == "all":
            return

        assert isinstance(fmax, (float, int)), "fmax must be a float or int"

        pad = int(round(self.sourceValue.shape[0]/2))
        n = self.sourceValue.shape[0] + 2 * pad

        tf = fftfreq(n, self.dt)
        y_fft = np.zeros((n,self.sourceValue.shape[1]), dtype="complex_")
        y = np.zeros(y_fft.shape, dtype="complex_")

        for i in range(y_fft.shape[1]):
            y_fft[pad:n-pad,i] = self.sourceValue[:,i]
            y_fft[:,i] = fft(y_fft[:,i])

        isup = np.where(tf>=fmax)[0]
        imax = np.where(tf[isup]>=fmax+1)[0][0]
        i1 = isup[0]
        i2 = isup[imax]

        iinf = np.where(tf<=-fmax)[0]
        imin = np.where(tf[iinf]<=-fmax-1)[0][-1]

        i3 = iinf[imin]
        i4 = iinf[-1]

        for i in range(y_fft.shape[1]):
            y_fft[i1:i2,i] = np.cos((isup[0:imax] - i1)/(i2-i1) * np.pi/2)**2 * y_fft[i1:i2,i]
            y_fft[i3:i4,i] = np.cos((iinf[imin:-1] - i4)/(i3-i4) * np.pi/2)**2 * y_fft[i3:i4,i]
            y_fft[i2:i3,i] = 0

        for i in range(y.shape[1]):
            y[:,i] = ifft(y_fft[:,i])

        it0 = int(round(abs(self.minTime/self.dt))) + pad

        d = int(round(1.2/fmax/self.dt)) # Using fmax instead of f0

        i1 = max(it0 - 4*d, 0)
        i2 = int(round(i1 + d/4))

        i4 = min(n,n - pad + 4*d)
        i3 = int(round(i4 - d/4))

        for i in range(y.shape[1]):
            y[i1:i2,i] = np.cos((np.arange(i1,i2) - i2)/(i2-i1) * np.pi/2)**2 * y[i1:i2,i]
            y[i3:i4,i] = np.cos((np.arange(i3,i4) - i3)/(i4-i3) * np.pi/2)**2 * y[i3:i4,i]
            y[max(i1-d,0):i1,i] = 0.0
            y[i4:min(i4+d,n),i] = 0.0

        t = np.arange(self.minTime-pad*self.dt, self.maxTime+pad*self.dt+self.dt/2, self.dt)

        # Update the source values and the time minimum time of the simulation
        self.updateSourceValue(np.real(y[max(i1-d,0):min(i4+d,n),:]))
        self.minTimeSim = t[max(i1-d,0)]
        self.maxfreq = fmax

    def updateParameters(self,
                         dt=None,
                         minTime=None,
                         maxTime=None,
                         dtSeismo=None,
                         dtWaveField=None,
                         sourceType=None,
                         sourceFreq=None,
                         Stype="modelling",
                         L=None,
                         Nxy=None,
                         K=None,
                         bound=None,
                         freesurface=False,
                         **kwargs):
        """
        Update the parameters of the solver.
        This method allows modifying the parameters of the solver after initialization.
        It is useful to update the parameters without reinitializing the solver.

        If the velocity model is changed, the solver should be reinitialized.
        """
        assert isinstance(dt, (float, type(None))), "dt must be a float or None"
        assert isinstance(minTime, (float, type(None))), "minTime must be a float or None"
        assert isinstance(maxTime, (float, type(None))), "maxTime must be a float or None"
        assert isinstance(dtSeismo, (float, type(None))), "dtSeismo must be a float or None"
        assert isinstance(dtWaveField, (float, type(None))), "dtWaveField must be a float or None"
        assert isinstance(sourceType, (str, type(None))), "sourceType must be a string or None"
        assert isinstance(sourceFreq, (float, type(None))), "sourceFreq must be a float or None"
        assert isinstance(L, (list, float, type(None))), "L must be a list, float, or None"
        assert isinstance(Nxy, (list, int, type(None))), "Nxy must be a list, int, or None"
        assert isinstance(K, (int, type(None))), "K must be an int or None"
        assert isinstance(bound, str), "bound must be a string"
        assert isinstance(freesurface, bool), "freesurface must be a boolean"
        assert isinstance(minTime, float), "minTime must be a float"
        assert isinstance(maxTime, float), "maxTime must be a float"
        assert Stype.lower() in ("modelling", "modeling", "inversion", "fwi"), \
            "Stype takes the values: 'modelling', 'modeling', 'inversion', or 'fwi'."
        if dt is not None:
            assert dt > 0, "dt must be bigger than 0"
            self.dt = dt
        if minTime is not None:
            self.minTime = minTime
            self.minTimeSim = minTime
        if maxTime is not None:
            self.maxTime = maxTime
            self.maxTimeSim = maxTime
        if dtSeismo is not None:
            # dtSeismo must be a multiple of dt
            if dt is not None:
                if dtSeismo > dt:
                    ratio = dtSeismo / dt
                    if not ratio.is_integer():
                        if ratio > 1:
                            dtSeismo = round(ratio) * dt
                        else:
                            dtSeismo = dt
                elif dtSeismo < dt:
                    dtSeismo = dt
            self.dtSeismo = dtSeismo
        if dtWaveField is not None:
            # dtWaveField must be bigger or equal to dtSeismo and also a multiple of dtSeismo
            if dtSeismo is not None:
                if dtWaveField > dtSeismo:
                    ratio = dtWaveField / dtSeismo
                    if not ratio.is_integer():
                        if ratio > 1:
                            dtWaveField = round(ratio) * dtSeismo
                        else:
                            dtWaveField = dtSeismo
                elif dtWaveField < dtSeismo:
                    dtWaveField = dtSeismo
            self.dtWaveField = dtWaveField
        if sourceType is not None:
            self.sourceType = sourceType
        if sourceFreq is not None:
            self.sourceFreq = sourceFreq
        if L is not None:
            self.L = L
        if Nxy is not None:
            self.Nxy = Nxy
        if K is not None:
            self.K = K
        if bound is not None:
            self.bound = bound
        if freesurface is not None:
            self.freesurface = freesurface
        self.Stype = Stype
        self.args = kwargs


    def computeSrcAndRcvConstants(self, sourcesCoords=[], receiversCoords=[]):

        # If lists are empty source and receivers constants are not updated.
        self.precomputeRcv(receiversCoords)
        self.precomputeSrc(sourcesCoords)


    def precomputeRcv(self, rcvCoords=[]):

        # If input is a numpy array, convert to list for uniformity
        if isinstance(rcvCoords, np.ndarray):
            rcvCoords = rcvCoords.tolist()
        # If input is a single pair (list or tuple of two scalars), wrap it
        if (
            isinstance(rcvCoords, (list, tuple))
            and len(rcvCoords) == 2
            and all(isinstance(x, (int, float, np.integer, np.floating)) for x in rcvCoords)
        ):
            rcvCoords = [rcvCoords]

        self.RcvNodes = np.zeros((len(rcvCoords),(self.subdomain.K+1)**2), dtype=np.int64)
        self.RcvNodalWeights = np.zeros((len(rcvCoords),(self.subdomain.K+1)**2), dtype=np.float64)
        for receiver_index, (xr, yr) in enumerate(rcvCoords):

            ixr = xr % self.subdomain.h
            ixr /= self.subdomain.h
            iyr = yr % self.subdomain.h
            iyr /= self.subdomain.h

            nodes = self.subdomain.getNodesFromPos(xr, yr)
            if len(nodes) != 0:
                self.RcvNodes[receiver_index,:] = nodes
                self.RcvNodalWeights[receiver_index,:] = lagrange_nodal_weights(self.K,ixr,iyr)
            else:
                continue


    def precomputeSrc(self, srcCoords):

        # If input is a numpy array, convert to list for uniformity
        if isinstance(srcCoords, np.ndarray):
            srcCoords = srcCoords.tolist()
        # If input is a single pair (list or tuple of two scalars), wrap it
        if (
            isinstance(srcCoords, (list, tuple))
            and len(srcCoords) == 2
            and all(isinstance(x, (int, float, np.integer, np.floating)) for x in srcCoords)
        ):
            srcCoords = [srcCoords]

        self.SrcNodes = np.zeros((len(srcCoords),(self.subdomain.K+1)**2), dtype=np.int64)
        self.SrcNodalWeights = np.zeros((len(srcCoords),(self.subdomain.K+1)**2), dtype=np.float64)
        for source_index, (xs, ys) in enumerate(srcCoords):

            # Map the coordinates to the relative element coordinates inside [0,1]
            ixs = xs % self.subdomain.h
            ixs /= self.subdomain.h
            iys = ys % self.subdomain.h
            iys /= self.subdomain.h

            nodes = self.subdomain.getNodesFromPos(xs,ys)
            if len(nodes) != 0:
                self.SrcNodes[source_index,:] = nodes
                self.SrcNodalWeights[source_index,:] = lagrange_nodal_weights(self.K,ixs,iys)
            else:
                continue


    def updateForwardSourceTerm(self, sourceval):
        """
        Constructs the source term vector by projecting the actual source value(s)
        onto the computational domain using Lagrange basis functions.

        Args:
            sourceval (float or nump.ndarray): The value(s) of the source to be applied at the nodes
            associated with the source position(s).

        """
        sourceval = np.asarray(sourceval)
        # First set the source term vector `F` to zero for all nodes associated with the source positions.
        for source_index in range(len(self.SrcNodes)):
            if self.SrcNodes[source_index] is not None:
                self.F[self.SrcNodes[source_index]] = 0.0
        for source_index in range(len(self.SrcNodes)):
            if self.SrcNodes[source_index] is not None:
                # Addition to account for nodes receiving the contribution of two sources (source offset <= node offset).
                self.F[self.SrcNodes[source_index]] += sourceval[source_index] * self.SrcNodalWeights[source_index]


    def updateBackwardSourceTerm(self, sourceval):
        """
        Computes the source term vector `F` for the adjoint modeling of a FWI process.
        Evidently, it can also be used in a RTM process.

        It calculates the source term vector `F` based on the reference source value(s) (`sourceval`),
        to be applied, at a given time, into the nodes associated with the receiver positions.

        Args:
            sourceval (float or nump.ndarray): The reference source value(s) of each seismic trace.
        """
        sourceval = np.asarray(sourceval)
        # First set the source term vector `F` to zero for all nodes associated with the source positions.
        for source_index in range(len(self.RcvNodes)):
            if self.RcvNodes[source_index] is not None:
                self.F[self.RcvNodes[source_index]] = 0.0
        for source_index in range(len(self.RcvNodes)):
            if self.RcvNodes[source_index] is not None:
                # Addition to account for nodes contributing to two receiver positions (receiver offset <= node offset).
                self.F[self.RcvNodes[source_index]] += sourceval[source_index] * self.RcvNodalWeights[source_index]


    def resetPressureAtReceivers(self, nxyr=1):
        """
        Resets the pressure values at receiver locations.

        This method reinitializes the array used to store the P-wave seismogram
        at receiver locations. Values are set to zero.

        Parameters:
        -----------
        nxyr : int, optional
            The number of receiver points (per shot) in the simulation. Default is 1.
        """

        Nt = (self.maxTime - self.minTime) / self.dtSeismo

        Nt = int(round(Nt)) + 1

        # Reset P-wave Seismogram
        self.PAtRecvs = np.zeros((Nt, nxyr), dtype=np.float64)


    def resetWaveFields(self, resetall=False):
        """
        Resets the wavefields used in the simulation.

        This method reinitializes the arrays representing the wavefields at
        different time steps (Pn0, Pn1, Pn2) to zero. If `resetall` is True,
        it also resets the adjoint wavefield (Padj) and the gradient (grad) to zero.
        """

        self.Pn0[:] = 0.0
        self.Pn1[:] = 0.0
        self.Pn2[:] = 0.0

        if self.Stype.lower() in ("inversion","fwi") and resetall:
            # Reset Adjoint wavefield
            self.Padj[:] = 0.0
            # Reset Gradient
            self.grad[:] = 0.0


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

        self.Pn0[:] = self.Pn1[:]
        self.Pn1[:] = self.Pn2[:]


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
        if flag.lower() not in ["forward", "backward"]:
            raise ValueError("flag must be 'forward' or 'backward'")

        if flag.lower() == "forward":
            for receiver_index in range(len(self.RcvNodes)):
                if self.RcvNodes[receiver_index] is not None:
                    self.PAtRecvs[timesample, receiver_index] = np.dot(self.Pn2[self.RcvNodes[receiver_index]],self.RcvNodalWeights[receiver_index])
        elif flag.lower() == "backward":
            for receiver_index in range(len(self.SrcNodes)):
                if self.SrcNodes[receiver_index] is not None:
                    self.PAtRecvs[timesample, receiver_index] = np.dot(self.Pn2[self.SrcNodes[receiver_index]],self.SrcNodalWeights[receiver_index])
        else:
            raise ValueError("flag must be 'forward' or 'backward'")


    def outputWavefield(self,refname="Pfield",directory="WaveFields",comm=None,typeWavefield="Full"):
        """
        Outputs the wavefield data to an HDF5 file.

        This method saves the wavefield data in a parallel I/O manner using the HDF5 format.
        Depending on the `typeWavefield` parameter, it can output either the full global wavefield
        or partial wavefield data for each MPI rank.

        Parameters:
        -----------
        refname : str, optional
            Reference name for the output file. Defaults to "Pfield".
        directory : str, optional
            Directory where the output file will be saved. Defaults to "WaveFields".
        comm : MPI.Comm, optional
            MPI communicator for parallel I/O. If None, the default communicator is used.
        typeWavefield : str, optional
            Specifies the type of wavefield to output. Options are:
            - "Full": Outputs the full global wavefield data.
            - "Partial": Outputs the partial wavefield data for each MPI rank.
        """
        rank = comm.Get_rank()

        if directory is None:
            directory = "WaveFields"

         # Create the output directory if it doesn't exist
        if rank == 0:
            os.makedirs(directory, exist_ok=True)

        comm.Barrier()  # Ensure all ranks wait until the directory is created
        if 'partial' in typeWavefield.lower():
            os.makedirs(f"{directory}/rank{rank}", exist_ok=True)

        if 'full' in typeWavefield.lower():

            globNodes = self.subdomain.getGlobalNodes(True)
            nodes = self.subdomain.getNodes(True)

            filename = f"./{directory}/{refname}.hdf5"
            h5f = h5py.File(filename, "w", driver="mpio", comm=comm)
            data = h5f.create_dataset("Pressure", data=np.zeros((self.subdomain.nxg*self.subdomain.nyg), dtype=np.float64))

            data[globNodes] = self.Pn2[nodes]
            h5f.close()

        elif 'partial' in typeWavefield.lower():

            # One filename per rank
            filename = f"./{directory}/rank{rank}/{refname}.hdf5"

            # Writting to the HDF5 file
            with h5py.File(filename, "w") as h5f:
                h5f.create_dataset("Pressure", data=self.Pn2, dtype=np.float64)

        else:
            if rank == 0:
                print("Warning: typeWavefield must be 'Full' or 'Partial'. No wavefield output.")


    def readWavefield(self,refname="Pfield",directory="WaveFields",comm=None,typeWavefield="Full"):
        """
        Reads the wavefield data from an HDF5 file.

        This method retrieves the wavefield data stored in HDF5 format. Depending on the
        `typeWavefield` parameter, it can read either the full global wavefield or partial
        wavefield data for each MPI rank.

        Parameters:
        -----------
        refname : str, optional
            Reference name for the input file. Defaults to "Pfield".
        directory : str, optional
            Directory where the input file is located. Defaults to "WaveFields".
        comm : MPI.Comm, optional
            MPI communicator for parallel I/O. If None, the method assumes a single process.
        typeWavefield : str, optional
            Specifies the type of wavefield to read (referring to how it was stored). Options are:
            - "Full": Reads the full global wavefield data.
            - "Partial": Reads the partial wavefield data for each MPI rank.

        Returns:
        --------
        numpy.ndarray
            The wavefield data read from the file. For "Full", it returns the global wavefield
            data for the current rank's subdomain. For "Partial", it returns the local wavefield
            data for the current rank.
        """
        rank = comm.Get_rank()

        if directory is None:
            directory = "WaveFields"

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        if 'full' in typeWavefield.lower():

            globNodes = self.subdomain.getGlobalNodes(True)
            nodes = self.subdomain.getNodes(True)

            filename = f"./{directory}/{refname}.hdf5"
            with h5py.File(filename, "r", driver="mpio", comm=comm) as h5f:
                data = h5f["Pressure"][:]

            # Create a numpy array to store the local data
            local_data = np.zeros_like(self.Pn2)
            local_data[nodes] = data[globNodes]

        elif 'partial' in typeWavefield.lower():

            # One filename per rank
            filename = f"./{directory}/rank{rank}/{refname}.hdf5"

            # Reading the HDF5 file
            with h5py.File(filename, "r") as h5f:
                local_data = h5f["Pressure"][:]

        else:
            if rank == 0:
                print("Warning: typeWavefield must be 'Full' or 'Partial'. No wavefield could be read.")

        return local_data


    def deleteWavefield(self,refname="Pfield",directory="WaveFields",comm=None,typeWavefield="Full"):
        """
        Deletes the wavefield stored as a HDF5 file.

        This method deletes the wavefield data stored in HDF5 format. Depending on the
        `typeWavefield` parameter, it will delete either the full global wavefield or the partial
        wavefield data for each MPI rank.

        Parameters:
        -----------
        refname : str, optional
            Reference name for the input file. Defaults to "Pfield".
        directory : str, optional
            Directory where the wavefield file is located. Defaults to "WaveFields".
        comm : MPI.Comm, optional
            MPI communicator for parallel I/O. If None, the method assumes a single process.
        typeWavefield : str, optional
            Specifies the type of wavefield to delete (referring to how it was stored). Options are:
            - "Full": Reads the full global wavefield data.
            - "Partial": Reads the partial wavefield data for each MPI rank.

        """
        rank = comm.Get_rank()

        if directory is None:
            directory = "WaveFields"

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"The directory '{directory}' does not exist.")

        if 'full' in typeWavefield.lower():

            filename = f"./{directory}/{refname}.hdf5"
            if os.path.isfile(filename):
                os.remove(filename)

        elif 'partial' in typeWavefield.lower():

            # One filename per rank
            filename = f"./{directory}/rank{rank}/{refname}.hdf5"

            if os.path.isfile(filename):
                os.remove(filename)

        else:
            if rank == 0:
                print("Warning: typeWavefield must be 'Full' or 'Partial'. No wavefield could be deleted.")


    def getPressureAtReceivers(self, comm=None):
        """
        Gather and broadcast pressure values at receivers across MPI ranks.

        In simpler terms it updates the pressure values at receiver locations
        across all MPI ranks. This way the value of solver.PAtRecvs is the same for all ranks.

        Parameters:
        -----------
        comm : MPI.Comm, optional
            MPI communicator for parallel execution. If None, the method assumes
            a single process and directly returns the local pressure values.

        Returns:
        --------
        numpy.ndarray
            The gathered and broadcasted pressure values at receivers.
        """
        if comm is None:
            # If no MPI communicator is provided, return local pressure values
            return self.PAtRecvs

        rank = comm.Get_rank()
        allPressure = comm.gather(self.PAtRecvs, root=0)
        pressure = None

        if rank == 0:

            pressure = np.zeros(self.PAtRecvs.shape)

            for p in allPressure:
                nonzero_columns = np.any(p, axis=0)
                pressure[:, nonzero_columns] = p[:, nonzero_columns]

        pressure = comm.bcast(pressure, root=0)

        return pressure


    def setModelParametrization(self, modelparametrization):
        """
        Set the parametrization of the model used in the solver.

        TODO: In the future density parametrization will be added.

        Parameters
        -----------
            model : str
                Model used for the solver. It can be 'c', '1/c', '1/c2'.
        """

        assert modelparametrization in ("c", "1/c", "1/c2"), "Model must be 'c', '1/c', or '1/c2'."
        self.modelparametrization = modelparametrization


    def updateModelfromFile(self, filename, lowvalue=None, highvalue=None, comm=None):
        """
        Update the model from a file. The file must be in HDF5 format.

        Parameters:
        -----------
        filename : str
            Path to the HDF5 file containing the velocity model. The file must
            include a dataset with the key "velocity".
        lowvalue : float, optional
            Minimum value to clip the velocity model to. If None, no lower bound
            clipping is applied. Must be greater than 0 if specified.
        highvalue : float, optional
            Maximum value to clip the velocity model to. If None, no upper bound
            clipping is applied. Must be greater than 0 if specified.
        comm : MPI.Comm, optional
            MPI communicator for parallel processing. If provided, the velocity
            model will be broadcasted to all processes. If None, only the root
            process (rank 0) will handle the file operations.

        Returns:
        --------
        numpy.ndarray
            The updated velocity model as a NumPy array.
        """
        if comm is None:
            rank = 0
        else:
            rank = comm.Get_rank()

        if rank == 0:
            x = None
            f = None
            try:
                f = h5py.File(filename, 'r')
                if "velocity" not in f:
                    raise KeyError("The key 'velocity' is missing in the HDF5 file.")
                x = f["velocity"][:]
            finally:
                if f is not None:
                    f.close()

            if x is not None:
                if lowvalue is not None and lowvalue > 0:
                    np.clip(x, lowvalue, None, out=x)

                if highvalue is not None and highvalue > 0:
                    np.clip(x, None, highvalue, out=x)

                if self.modelparametrization == "1/c2":
                    x = np.sqrt(1/x)
                elif self.modelparametrization == "1/c":
                    x = 1/x
                elif self.modelparametrization == "c":
                    pass
            else:
                raise ValueError("The velocity model is empty or not found in the file.")
        else:
            x = None

        if comm is not None:
            x = comm.bcast( x, root=0 )

        self.velocity = x[self.subdomain.globElems]


    def updateGradient(self, cycle, shotId, comm=None, typeWavefield="Full"):
        """
        Updates the gradient for the inversion or FWI process.

        This method calls the `computeGradient` function to calculate the gradient based on the
        current wavefield and the adjoint wavefield. The gradient is updated in-place
        in the `grad` attribute of the solver.

        Parameters:
        -----------
        cycle : int
            The current cycle of the inversion or FWI process.
        shotId : int
            The shot identifier for which the gradient is being updated.
        comm : MPI.Comm, optional
            MPI communicator for parallel execution. If None, the method assumes a single process.
        typeWavefield : str, optional
            Specifies the type of wavefield to read. Options are:
            - "Full": Reads the full global wavefield data.
            - "Partial": Reads the partial wavefield data for each MPI rank.

        Notes:
        ------
        - This method is only applicable in "inversion" or "fwi" modes.
        - This method should be called during a backward pass of the simulation.
        """
        assert isinstance(self.Stype, str) and self.Stype.lower() in ("inversion", "fwi"), f"Gradient can only be updated in inversion or fwi mode. Current mode: {self.Stype}"

        self.Padj = self.readWavefield(refname="P_cycle"+str(cycle)+"_shot"+str(shotId), directory="ForwardWavefields", comm=comm, typeWavefield=typeWavefield)
        # Delete the wavefield after reading it to avoid storing it in memory.
        self.deleteWavefield(refname="P_cycle"+str(cycle)+"_shot"+str(shotId), directory="ForwardWavefields", comm=comm, typeWavefield=typeWavefield)

        self.grad[:] = computeGradient(self.grad, self.Pn2, self.Pn1, self.Pn0, self.dt, self.Mterm, self.Padj, self.subdomain.Ix, self.subdomain.Iy, self.K)


    def storeGradientInFile(self, shotId, comm, gradDirectory="partialGradient"):
        """
        Compute and store the gradient for each shot in an HDF5 file.

        This method gathers the gradient data from all MPI ranks, and combines it into a global gradient.
        The resulting gradient is stored in an HDF5 file named according to the shotId.
        The file is saved in the specified directory.

        Parameters
        ----------
        shotId : str
            Identifier for the shot, used to name the output file.
        comm : MPI.Comm
            MPI communicator for parallel execution.
        gradDirectory : str, optional
            Directory where the partial gradient files will be stored.
            Default is `partialGradient`.

        Notes
        -----
        - The method ensures that the gradient data is synchronized across all MPI ranks.
        - The gradient file is first written with a temporary name and then renamed to indicate readiness.
        - The directory for storing the gradient files is created if it does not already exist.
        """
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = 0

        if self.Stype.lower() not in ("inversion", "fwi"):
            if rank == root:
                print("Gradient storage is only applicable in inversion or fwi mode. Skipping.")
            return

        # Gather global gradient
        globElems = self.subdomain.getGlobalElems(True) # Global elements relative to each process
        elements = self.subdomain.getElems(True) # Local elements inside subdomain
        rGElems = comm.gather(globElems,root=root) # Map global elements position of all processes to rank==0
        gradFAN = comm.gather(self.grad[elements],root=root) # Gradient from all elements

        if rank == root:

            gradFull = np.zeros((self.subdomain.Ixg*self.subdomain.Iyg), dtype=np.float64)

            for i in range(size):
                gradFull[rGElems[i]] = gradFAN[i]

            os.makedirs(gradDirectory, exist_ok=True)

            with h5py.File(f"{gradDirectory}/partialGradient_"+shotId+".hdf5", 'w') as h5p:

                h5p.create_dataset("partialGradient", data=self.dtWaveField * gradFull,
                                   chunks=True, maxshape=(self.subdomain.Ixg*self.subdomain.Iyg,))
                # Scaling the gradient by the dtWaveField is necessary to ensure the correct values when summing the gradients.

            shutil.move(f"{gradDirectory}/partialGradient_"+shotId+".hdf5", f"{gradDirectory}/partialGradient_ready_"+shotId+".hdf5")

        comm.Barrier()


    def initspongeTaper(self, spongeWidth=0.1, spongeFactor=1.0, inUpper=False, vMax=0.0, exportTaper=False, comm=None):
        """
        Calculates the sponge layer to apply to the wavefield in order to
        further attenuate outgoing waves.

        !!!!!! For the moment same size to both dimension: need to finish implementing

        Parameters:
        -----------
        spongeWidth : float, optional
            The width of the sponge layer as a fraction of the total domain size.
            Default is 0.1 (10% of the total domain size).
            Upper part of the domain is not considered by default.
        spongeFactor : float, optional
            The factor by which the sponge layer attenuates the wavefield.
            Default is 1.0 (no attenuation).
        inUpper : bool, optional
            If True, the sponge layer is applied to the upper part of the domain.
            Default is False.
        vMax : float, optional
            The maximum velocity used to calculate the sponge layer.
            Default is 0.0 (no taper).
        exportTaper : bool, optional
            If True, the taper values are exported to a hdf5 file.
            Default is False (no export).
        comm : MPI.Comm, optional
            MPI communicator for parallel execution. If None, the method assumes
            a single process.
        """
        assert 0 < spongeWidth < 1, "spongeWidth must be a positive float <1"

        globNodes = self.subdomain.getGlobalNodes(True)
        nodes = self.subdomain.getNodes(True)

        spongeLx = int(round(spongeWidth * self.subdomain.nxg))
        spongeLy = int(round(spongeWidth * self.subdomain.nyg))

        l_idx = 0
        for a in globNodes:
            a_i = a // self.subdomain.nxg # row index
            a_j = a % self.subdomain.nxg # column index
            distXmin = a_j
            distXmax = self.subdomain.nxg - a_j
            distYmax = self.subdomain.nyg - a_i
            dist = min(distXmin, distXmax, distYmax)
            if inUpper:
                distYmin = a_i
                dist = min(dist, distYmin)
            else:
                distYmin = 2*distYmax # Value just need to be bigger than distYmax

            distXmin = min(distXmin,distXmax)
            distYmin = min(distYmin,distYmax)
            if dist < spongeLx:
                self.Taper[nodes[l_idx]] = np.exp((((3*vMax)/(2*spongeLx))*np.log( spongeFactor )*pow((spongeLx-dist)/spongeLx, 2 ))*self.dt )

            l_idx += 1

        if exportTaper:

            if comm is None:
                rank = 0
            else:
                rank = comm.Get_rank()

            # Create the output directory if it doesn't exist
            if rank == 0:
                os.makedirs("TaperDBG", exist_ok=True)

            if comm is not None:
                comm.Barrier()  # Ensure all ranks wait until the directory is created

            if comm is not None:
                h5f = h5py.File("TaperDBG/spongeTaper.hdf5", "w", driver="mpio", comm=comm)
            else:
                h5f = h5py.File("TaperDBG/spongeTaper.hdf5", "w")

            data = h5f.create_dataset("TaperValues", data=np.zeros((self.subdomain.nxg*self.subdomain.nyg), dtype=np.float64))
            data[globNodes] = self.Taper[nodes]
            h5f.close()

            if comm is not None:
                comm.Barrier()
