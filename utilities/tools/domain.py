import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


class Domain:
    """
    Domain Class
    The `Domain` class represents a 2D computational domain for numerical simulations.
    It provides methods to define the domain's geometry, boundary conditions, and
    mapping between local and global indices for degrees of freedom (DOFs). The class
    ensures that the elements in the domain are square and supports both single and
    multi-dimensional configurations.
    Attributes:
        Lx (float): Length (in meters) of the domain in the x-direction.
        Ix (int): Number of elements in the x-direction.
        Ly (float, optional): Length (in meters) of the domain in the y-direction. Defaults to `Lx`.
        Iy (int, optional): Number of elements in the y-direction. Defaults to `Ix`.
        K (int): Number of local basis functions per element in one direction. Defaults to 1.
        boundary_conditions (str): Type of boundary conditions. Defaults to "neumann".
        h (float): Size of each square element.
        nx (int): Number of nodes in the x-direction.
        ny (int): Number of nodes in the y-direction.
        boundaryNodes (numpy.ndarray): Array of indices representing boundary nodes.
    """

    def __init__(self, Lx, Ix, Ly=None, Iy=None, K=1, boundary_conditions="neumann", freesurface=False):

        self.Lx = Lx
        self.Ix = int(Ix)
        self.K = K
        self.boundary_conditions = boundary_conditions
        self.freesurface = freesurface

        if Ly is not None:
            self.Ly = Ly
        else:
            self.Ly = Lx

        if Iy is not None:
            self.Iy = int(Iy)
        else:
            self.Iy = int(Ix)

        if abs((self.Lx/self.Ix) - (self.Ly/self.Iy))/(self.Lx/self.Ix) >= 1e-8:
            raise ValueError("Elements must be square")
        else:
            self.h = self.Lx/self.Ix

        self.nx = int(Ix*self.K + 1)
        self.ny = int(self.Iy*self.K + 1)

        self.boundaryNodes, self.upperNodes = self.getBoundNodes()


    def getBoundNodes(self):
        """
        Get the boundary nodes of a 2D domain.

        This method calculates and returns the indices of the nodes that lie on the
        boundaries of a 2D computational domain. The boundaries include the left,
        right, top (up), and bottom (down) edges of the domain.

        Returns:
            numpy.ndarray: An array containing the unique indices of the boundary
            nodes in the domain.
        """

        Ix = self.Ix
        Iy = self.Iy
        nx = self.nx
        ny = self.ny
        K = self.K

        left = np.arange(0, nx).tolist() # top nodes?
        right = np.arange(nx*Iy*K, nx*ny).tolist() # bottom nodes?
        down = np.arange(Ix*K, nx*ny, nx).tolist() # rightmost nodes?
        up = np.arange(0, nx*Iy*K+1, nx).tolist() # leftmost nodes?

        boundNodes = np.array(list(set(up + down + right + left)))

        return boundNodes, left

    def loc2loc(self,k1,k2):
        """
        Associates the two local indices k1 and k2 (in x and y directions) of a local degree of freedom (ddl)
        to a single local index for the degree of freedom.
        k1 and k2 are the local numbers of the basis function in x and y directions.
        """
        K = self.K

        return(k1+(K+1)*k2)

    def loc2glob(self, i1, i2, k1, k2):
        """
        Converts local indices of a basis function to a global degree of freedom (DOF) index.

        Args:
            i1 (int): Element index along the x-direction.
            i2 (int): Element index along the y-direction.
            k1 (int): Local basis function index along the x-direction.
            k2 (int): Local basis function index along the y-direction.

        Returns:
            int: Global degree of freedom (DOF) index corresponding to the given local indices.

        Notes:
            - `nx` represents the number of elements along the x-direction.
            - `K` represents the number of local basis functions per element in one direction.
            - The formula calculates the global index by combining the local indices and element indices.
        """

        nx = self.nx
        K = self.K

        m = int(nx*(i2*K+k2)+i1*K+k1)

        return m

    def ele2ele(self,i1,i2):
        """
        Maps two indices (i1, i2) to a unique element number.
        This function calculates a unique element number based on the provided
        indices `i1` and `i2`, which represent coordinates in the x and y
        directions, respectively. The mapping is performed using the `Ix`
        attribute, which represents the number of elements in the x direction.
        Args:
            i1 (int): The index in the x direction.
            i2 (int): The index in the y direction.
        Returns:
            int: The unique element number corresponding to the given indices.
        """

        Ix = self.Ix

        e = int(i1+Ix*i2)

        return e

    def getElemIndexes(self, e):
        """
        Calculate the 2D grid element indices (i, j) from a 1D element index.

        This method converts a 1D index `e` into its corresponding 2D grid
        indices `(i, j)` based on the number of elements in the x-direction (`Ix`).

        Args:
            e (int): The 1D element index.

        Returns:
            tuple: A tuple `(i, j)` where `i` is the column index and `j` is the row index.
        """
        Ix = self.Ix
        j = e//Ix
        i = e%Ix

        return i,j

class Subdomain(Domain):

    def __init__(self, domain, mpi):

        self.Ixg = domain.Ix
        self.Iyg = domain.Iy
        self.nxg = domain.nx
        self.nyg = domain.ny

        boundary_conditions = domain.boundary_conditions

        self.freesurface = domain.freesurface

        pos_x = mpi.pos_x
        pos_y = mpi.pos_y

        Px = mpi.proc_x
        Py = mpi.proc_y

        ghost_layers_up_down = mpi.ghost_layers_up_down
        ghost_layers_left_right = mpi.ghost_layers_left_right

        base_size_x = self.Ixg // Px  # Taille de base pour chaque processus
        remainder_x = self.Ixg % Px  # Nombre de processus qui recevront un élément supplémentaire

        base_size_y = self.Iyg // Py  # Taille de base pour chaque processus
        remainder_y = self.Iyg % Py  # Nombre de processus qui recevront un élément supplémentaire
        # La taille de chaque sous-domaine : ceux avec remainder > 0 recevront un élément en plus
        diff_x = remainder_x
        sizes_x = np.zeros(Px,int) + base_size_x

        diff_y = remainder_y
        sizes_y = np.zeros(Py,int) + base_size_y
        for i in range(remainder_x):
            if diff_x != 0:
                sizes_x[i] += 1
                diff_x -=1
            else:
                break
        for i in range(remainder_y):
            if diff_y != 0:
                sizes_y[i] += 1
                diff_y -=1
            else:
                break

        offsets_x = np.cumsum([0] + sizes_x[:-1].tolist())# Calcul des décalages
        offsets_y = np.cumsum([0] + sizes_y[:-1].tolist())

        sizes = [sizes_x[pos_x] + ghost_layers_up_down, sizes_y[pos_y]+ghost_layers_left_right]
        Ix = sizes[0]
        Iy = sizes[1]
        Lx = Ix * domain.h
        Ly = Iy * domain.h
        K = domain.K

        self.offsets = [offsets_x[pos_x], offsets_y[pos_y]]

        super().__init__(Lx, Ix, Ly, Iy, K, boundary_conditions)

        self.globElems = self.getGlobalElems()
        self.globNodes = self.getGlobalNodes()

        boundaryNodes = []
        upperNodes = []
        for i1 in range(Ix):
            for i2 in range(Iy):
                for k1 in range(K+1):
                    for k2 in range(K+1):
                        k = self.loc2glob(i1,i2,k1,k2)
                        kg = self.subNode2Node(i1,i2,k1,k2)
                        if kg in domain.boundaryNodes:
                            boundaryNodes.append(k)
                        if self.freesurface and kg in domain.upperNodes:
                            upperNodes.append(k)

        self.boundaryNodes = list(set(boundaryNodes))
        if self.freesurface:
            self.upperNodes = list(set(upperNodes))

        self.set_ghostNodes(mpi)

        self.nodesList = self.getNodes(True)
        self.elemsList = self.getElems(True)

    def isInsideDomain(self, x, y):
        h = self.h
        Ixg = self.Ixg

        i1 = int(x/h)
        i2 = int(y/h)

        ig = i1+Ixg*i2

        if ig in self.globElems:
            if self.offsets[1] == 0:
                iy = ig//self.Ixg - self.offsets[1]
            else:
                iy = ig//self.Ixg - self.offsets[1] + 1
            if self.offsets[0] == 0:
                ix = ig%self.Ixg - self.offsets[0]
            else:
                ix = ig%self.Ixg - self.offsets[0] + 1
            return True, ix, iy
        else:
            return False, -1, -1


    def subElem2Elem(self,i1,i2):
        Ixg = self.Ixg

        offsets = self.offsets

        # i1 et i2 numero suivant x et y de l element
        if offsets[0] == 0:
            ig1 = offsets[0] + i1
        else:
            ig1 = offsets[0]-1 + i1
        if offsets[1] == 0:
            ig2 = offsets[1] + i2
        else:
            ig2 = offsets[1]-1 + i2

        ig = ig1 + ig2 * Ixg

        return ig


    def subNode2Node(self,i1,i2,k1,k2):
        nxg = self.nxg
        K = self.K
        offsets = self.offsets
        # i1 et i2 numero suivant x et y de l element
        if offsets[0] == 0:
            ig1 = offsets[0] + i1
        else:
            ig1 = offsets[0]-1 + i1
        if offsets[1] == 0:
            ig2 = offsets[1] + i2
        else:
            ig2 = offsets[1]-1 + i2

        ng = nxg*(ig2*K+k2)+ig1*K+k1

        return ng


    def getElems(self, filterGhost=False):
        Ix = self.Ix
        Iy = self.Iy

        elems = []
        for i1 in range(Ix):
            for i2 in range(Iy):
                e = self.ele2ele(i1,i2)


                if filterGhost:
                    if e not in self.ghostElems:
                        elems.append(e)
                else:
                    elems.append(e)

        return np.sort(np.array(list(set(elems)),int))


    def getGlobalElems(self, filterGhost=False):
        Ix = self.Ix
        Iy = self.Iy

        globElems = []
        for i1 in range(Ix):
            for i2 in range(Iy):
                if filterGhost:
                    if self.ele2ele(i1,i2) not in self.ghostElems:
                        globElems.append(self.subElem2Elem(i1,i2))
                else:
                    globElems.append(self.subElem2Elem(i1,i2))

        return np.sort(np.array(list(set(globElems)),int))


    def getNodes(self, filterGhost=False):
        Ix = self.Ix
        Iy = self.Iy
        K = self.K

        nodes = []
        for i1 in range(Ix):
            for i2 in range(Iy):
                for k1 in range(K+1):
                    for k2 in range(K+1):
                        k = self.loc2glob(i1,i2,k1,k2)
                        if filterGhost:
                            if k not in self.ghostNodes:
                                nodes.append(k)
                        else:
                            nodes.append(k)

        return np.sort(np.array(list(set(nodes)),int))


    def getNodesFromPos(self, x, y):
        """
        Get the local node indices corresponding to a position (x, y) in the subdomain.
        This method checks if the position is inside the subdomain and returns the
        local node indices if it is. If the position is outside the subdomain, it returns an empty list.
        """
        K = self.K
        isInside, i1, i2 = self.isInsideDomain(x,y)
        if isInside:
            nodes = np.zeros((K+1)**2, int)
            for k1 in range(K+1):
                for k2 in range(K+1):
                    k = self.loc2glob(i1,i2,k1,k2)
                    nodes[k2 + (K+1)*k1] = k

            return nodes
        else:
            return []


    def getGlobalNodes(self, filterGhost=False):
        Ix = self.Ix
        Iy = self.Iy
        K = self.K

        globNodes = []
        for i1 in range(Ix):
            for i2 in range(Iy):
                for k1 in range(K+1):
                    for k2 in range(K+1):
                        k = self.loc2glob(i1,i2,k1,k2)
                        if filterGhost:
                            if k not in self.ghostNodes:
                                globNodes.append(self.subNode2Node(i1,i2,k1,k2))
                        else:
                            globNodes.append(self.subNode2Node(i1,i2,k1,k2))

        return np.sort(np.array(list(set(globNodes)),int))


    def set_ghostNodes(self, mpiDecomposition):
        K = self.K
        Ix = self.Ix
        Iy = self.Iy
        neighbors = mpiDecomposition.neighbors
        pos_x = mpiDecomposition.pos_x
        pos_y = mpiDecomposition.pos_y
        proc_x = mpiDecomposition.proc_x
        proc_y = mpiDecomposition.proc_y

        nx = self.nx
        ny = self.ny

        gn_l = []
        gn_r = []
        gn_u = []
        gn_d = []
        gn_ul = []
        gn_ur = []
        gn_dl = []
        gn_dr = []

        ghost_elems = []

        send_ul = []
        send_ur = []
        send_dl = []
        send_dr = []

        if neighbors["down"] == MPI.PROC_NULL:
            Ixl=Ix
        else:
            Ixl=Ix-1
        if neighbors["right"] == MPI.PROC_NULL:
            Iyl=Iy
        else:
            Iyl=Iy-1
        if neighbors["up"] == MPI.PROC_NULL:
            Ix0=-1
        else:
            Ix0=0
        if neighbors["left"] == MPI.PROC_NULL:
            Iy0=-1
        else:
            Iy0=0

        if neighbors["down"] != MPI.PROC_NULL:
            ghost_elems.append(np.arange(Ixl, Ix*Iy, Ix).tolist())
        if neighbors["right"] != MPI.PROC_NULL:
            ghost_elems.append(np.arange(Ix*Iyl, Ix*Iy).tolist())
        if neighbors["left"] != MPI.PROC_NULL:
            ghost_elems.append(np.arange(0, Ix).tolist())
        if neighbors["up"] != MPI.PROC_NULL:
            ghost_elems.append(np.arange(0, Ix*Iyl+1, Ix).tolist())

        if len(ghost_elems)!=0:
            self.ghostElems = np.array(list(set(np.concatenate(ghost_elems))),int)
        else:
            self.ghostElems = []

        for i1 in range(Ix):
            for i2 in range(Iy):
                for k1 in range(K+1):
                    for k2 in range(K+1):
                        k = self.loc2glob(i1,i2,k1,k2)

                        if pos_x==0 and pos_y==0:
                            if neighbors["down-right"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2==Iy-1 and (k1>0 and k2>0):
                                    gn_dr.append(k)
                                if i1==Ix-2 and i2==Iy-2:
                                    send_dr.append(k)
                            if neighbors["down"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2<Iyl and k1>0:
                                    gn_d.append(k)
                            if neighbors["right"] != MPI.PROC_NULL:
                                if i1<Ixl and i2==Iy-1 and k2>0:
                                    gn_r.append(k)

                        elif pos_x==proc_x-1 and pos_y==0:
                            if neighbors["up-right"] != MPI.PROC_NULL:
                                if i1==0 and i2==Iy-1 and k2>0:#(k1<K and k2>0):
                                    gn_ur.append(k)
                                if i1==1 and i2==Iy-2 and k1>0:#(k1!=0 and k2!=K):
                                    send_ur.append(k)
                            if neighbors["up"] != MPI.PROC_NULL:
                                if i1==0 and i2<Iyl:# and k1<K:
                                    gn_u.append(k)
                            if neighbors["right"] != MPI.PROC_NULL:
                                #if i1>Ix0 and i2==Iy-1 and k2>0:
                                if i1>Ix0 and i2==Iy-1 and k2>0 and k not in gn_ur:
                                    gn_r.append(k)

                        elif pos_x==0 and pos_y==proc_y-1:
                            if neighbors["down-left"] != MPI.PROC_NULL:
                                #if i1==Ix-1 and i2==0 and (k1>0 and k2<K):
                                if i1==Ix-1 and i2==0 and k1>0:
                                    gn_dl.append(k)
                                if i1==Ix-2 and i2==1 and k2>0:
                                    send_dl.append(k)
                            if neighbors["left"] != MPI.PROC_NULL:
                                if i1<Ixl and i2==0:# and k2<K:
                                    gn_l.append(k)
                            if neighbors["down"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2>Iy0 and k1>0 and k not in gn_dl:
                                    gn_d.append(k)

                        elif pos_x==proc_x-1 and pos_y==proc_y-1:
                            if neighbors["up-left"] != MPI.PROC_NULL:
                                if i1==0 and i2==0:
                                    gn_ul.append(k)
                                if i1==1 and i2==1 and (k1>0 and k2>0):
                                    send_ul.append(k)
                            if neighbors["left"] != MPI.PROC_NULL:
                                if i1>Ix0 and i2==0 and k not in gn_ul:
                                    gn_l.append(k)
                            if neighbors["up"] != MPI.PROC_NULL:
                                if i1==0 and i2>Iy0 and k not in gn_ul:
                                    gn_u.append(k)

                        elif pos_x==0 and 0<pos_y<proc_y-1:
                            if neighbors["down-left"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2==0 and k1>0:
                                    gn_dl.append(k)
                                if i1==Ix-2 and i2==1 and k2>0:
                                    send_dl.append(k)
                            if neighbors["down-right"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2==Iy-1 and (k1>0 and k2>0):
                                    gn_dr.append(k)
                                if i1==Ix-2 and i2==Iy-2:
                                    send_dr.append(k)

                            if neighbors["down"] != MPI.PROC_NULL:
                                if i1==Ix-1 and Iy0<i2<Iyl and k1>0 and k not in gn_dl:
                                    gn_d.append(k)

                            if i1<Ixl and i2==Iy-1 and k2>0:
                                gn_r.append(k)
                            elif i1<Ixl and i2==0:
                                gn_l.append(k)

                        elif pos_x==proc_x-1 and 0<pos_y<proc_y-1:
                            if neighbors["up-left"] != MPI.PROC_NULL:
                                if i1==0 and i2==0:
                                    gn_ul.append(k)
                                if i1==1 and i2==1 and (k1>0 and k2>0):
                                    send_ul.append(k)
                            if neighbors["up-right"] != MPI.PROC_NULL:
                                if i1==0 and i2==Iy-1 and k2>0:
                                    gn_ur.append(k)
                                if i1==1 and i2==Iy-2 and k1>0:
                                    send_ur.append(k)

                            if neighbors["up"] != MPI.PROC_NULL:
                                if i1==0 and Iy0<i2<Iyl and k not in gn_ul:
                                    gn_u.append(k)

                            if i1>Ix0 and i2==Iy-1 and k2>0 and k not in gn_ur:
                                gn_r.append(k)
                            elif i1>Ix0 and i2==0 and k not in gn_ul:
                                gn_l.append(k)

                        elif 0<pos_x<proc_x-1 and pos_y==0:
                            if neighbors["up-right"] != MPI.PROC_NULL:
                                if i1==0 and i2==Iy-1 and k2>0:
                                    gn_ur.append(k)
                                if i1==1 and i2==Iy-2 and k1>0:
                                    send_ur.append(k)

                            if neighbors["down-right"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2==Iy-1 and (k1>0 and k2>0):
                                    gn_dr.append(k)
                                if i1==Ix-2 and i2==Iy-2:
                                    send_dr.append(k)

                            if neighbors["right"] != MPI.PROC_NULL:
                                if Ix0<i1<Ixl and i2==Iy-1 and k2>0 and k not in gn_ur:
                                    gn_r.append(k)

                            if i1==0 and i2<Iyl:
                                gn_u.append(k)
                            elif i1==Ix-1 and i2<Iyl and k1>0:
                                gn_d.append(k)

                        elif 0<pos_x<proc_x-1 and pos_y==proc_y-1:
                            if neighbors["up-left"] != MPI.PROC_NULL:
                                if i1==0 and i2==0:
                                    gn_ul.append(k)
                                if i1==1 and i2==1 and (k1>0 and k2>0):
                                    send_ul.append(k)
                            if neighbors["down-left"] != MPI.PROC_NULL:
                                if i1==Ix-1 and i2==0 and k1>0:
                                    gn_dl.append(k)
                                if i1==Ix-2 and i2==1 and k2>0:
                                    send_dl.append(k)

                            if neighbors["left"] != MPI.PROC_NULL:
                                if Ix0<i1<Ixl and i2==0 and k not in gn_ul:
                                    gn_l.append(k)

                            if i1==0 and i2>Iy0 and k not in gn_ul:
                                gn_u.append(k)
                            elif i1==Ix-1 and i2>Iy0 and k1>0 and k not in gn_dl:
                                gn_d.append(k)

                        elif 0<pos_x<proc_x-1 and 0<pos_y<proc_y-1:
                            if i1==0 and i2==0:
                                gn_ul.append(k)
                            elif i1==0 and i2==Iy-1 and k2>0:
                                gn_ur.append(k)
                            elif i1==Ix-1 and i2==0 and k1>0:
                                gn_dl.append(k)
                            elif i1==Ix-1 and i2==Iy-1 and (k1>0 and k2>0):
                                gn_dr.append(k)
                            elif Ix0<i1<Ixl and i2==0 and k not in gn_ul:
                                gn_l.append(k)
                            elif i1==0 and Iy0<i2<Iyl and k not in gn_ul:
                                gn_u.append(k)
                            elif i1==Ix-1 and Iy0<i2<Iyl and k1>0 and k not in gn_dl:
                                gn_d.append(k)
                            elif Ix0<i1<Ixl and i2==Iy-1 and k2>0 and k not in gn_ur:
                                gn_r.append(k)
                            if i1==1 and i2==1 and (k1>0 and k2>0):
                                send_ul.append(k)
                            elif i1==Ix-2 and i2==Iy-2:
                                send_dr.append(k)
                            elif i1==1 and i2==Iy-2 and k1>0:
                                send_ur.append(k)
                            elif i1==Ix-2 and i2==1 and k2>0:
                                send_dl.append(k)

        ghost_nodes = {}
        ghost_nodes['left'] = np.sort(np.array(list(set(gn_l))))
        ghost_nodes['right'] = np.sort(np.array(list(set(gn_r))))
        ghost_nodes['up'] = np.sort(np.array(list(set(gn_u))))
        ghost_nodes['down'] = np.sort(np.array(list(set(gn_d))))
        ghost_nodes['up-right'] = np.sort(np.array(list(set(gn_ur))))
        ghost_nodes['up-left'] = np.sort(np.array(list(set(gn_ul))))
        ghost_nodes['down-right'] = np.sort(np.array(list(set(gn_dr))))
        ghost_nodes['down-left'] = np.sort(np.array(list(set(gn_dl))))

        send_nodes = {}
        nbl = int(len(ghost_nodes['left'])/(K+1))
        nbr = int(len(ghost_nodes['right'])/K)
        nbu = len(ghost_nodes['up'])
        nbd = len(ghost_nodes['down'])

        send_nodes['left'] = np.array([],int)
        for i in range(1,K+1):
            send_nodes['left'] = np.append(send_nodes['left'], ghost_nodes['left'][-nbl::] + i*nx)
        send_nodes['left'] = np.sort(send_nodes['left'])

        send_nodes['up'] = np.array([],int)
        for i in range(1,K+1):
            send_nodes['up'] = np.append(send_nodes['up'], ghost_nodes['up'][K:nbu:K+1] + i)
        send_nodes['up'] = np.sort(send_nodes['up'])

        send_nodes['right'] = np.array([],int)
        for i in range(1,K+2):
            send_nodes['right'] = np.append(send_nodes['right'], ghost_nodes['right'][0:nbr] - i*nx)
        send_nodes['right'] = np.sort(send_nodes['right'])

        send_nodes['down'] = np.array([],int)
        for i in range(1,K+2):
            send_nodes['down'] = np.append(send_nodes['down'], ghost_nodes['down'][0:nbd:K] - i)
        send_nodes['down'] = np.sort(send_nodes['down'])

        send_nodes['up-right'] = np.sort(np.array(list(set(send_ur))))
        send_nodes['up-left'] = np.sort(np.array(list(set(send_ul))))
        send_nodes['down-right'] = np.sort(np.array(list(set(send_dr))))
        send_nodes['down-left'] = np.sort(np.array(list(set(send_dl))))

        self.ghost_nodes = ghost_nodes
        self.send_nodes = send_nodes

        if len(ghost_nodes.values())!=0:
               self.ghostNodes = np.sort(np.concatenate(list(ghost_nodes.values()))).astype(int)
        else:
               self.ghostNodes = []
