import numba
import numpy as np
import scipy.sparse as sp
"""2D Spectral Element Method (SEM) Kernel.

This module provides the basis for solving 2D spectral element method (SEM) problems. 

Only acoustic wave equations are considered in this module.

The key features of this module include:
- Construction of global mass, stiffness, and damping matrices for SEM solvers.
- Implementation of Lagrange basis functions and their derivatives.
- Computation of Gauss-Lobatto quadrature points and weights for numerical integration.
Functions:
-----------
- delta(i, j):
    Computes the Kronecker delta, which is 1 if the two variables are equal, and 0 otherwise.
- lagrange_basis(x, i, nodes):
    Computes the Lagrange polynomial L_i(x) used in interpolation.
- derivative_lagrange_basis(x, i, nodes):
- gauss_lobatto_points_weights(n):
    Computes the Gauss-Lobatto quadrature points and weights on the interval [0, 1].
- buildMassAndStiffnessSEM(domain, model):
    Constructs the global mass and stiffness matrices for a 2D SEM solver.
- buildDampingSEM(domain, model):
    Constructs the global damping matrix for a 2D SEM solver.
-----------
- domain: 
    An object containing domain properties such as element size, boundary conditions, and node mappings.
- model: 
    An object containing material properties such as velocity and density.
--------
- M: 
    Sparse global mass matrix.
- R: 
    Sparse global stiffness matrix.
- S: 
    Sparse global damping matrix.
--------
Authors:
- Author(s): Julien Besset (Major contributor), Victor Martins Gomes (Minor contributor)
- Institution: INRIA
- Contact Information: julien.besset@inria.fr
Date:
-----
- Created: 2024
- Last Modified: April 2025
-------------------------------------------
"""


@numba.njit
def delta(i, j):
    """
    Computes the Kronecker delta, which is a function of two variables 
    (i and j) that is 1 if the variables are equal, and 0 otherwise.
    
    Parameters:
        i (int): The first variable to compare.
        j (int): The second variable to compare.
    
    Returns:
        int: 1 if i equals j, otherwise 0.
    """
    if i == j:
        return 1
    else:
        return 0


@numba.njit
def lagrange_basis(x, i, nodes):
    """
    Computes the Lagrange polynomial L_i(x), which is used in interpolation.

    Parameters:
        x (float): The evaluation point.
        i (int): The index of the Lagrange basis polynomial.
        nodes (list or np.array): The interpolation nodes.

    Returns:
        float: The value of the Lagrange polynomial L_i(x) at the given point.
    """
    L_i = 1
    for j in range(len(nodes)):
        if j != i:
            L_i *= (x - nodes[j]) / (nodes[i] - nodes[j])
    return L_i


@numba.njit
def derivative_lagrange_basis(x, i, nodes):
    """
    Computes the derivative of the Lagrange polynomial L'_i(x).

    Parameters:
        x (float): The evaluation point.
        i (int): The index of the Lagrange basis polynomial.
        nodes (list or np.array): The interpolation nodes.

    Returns:
        float: The value of the derivative of the Lagrange polynomial L'_i(x) at the given point.
    """
    dLdx_i = 0
    for j in range(len(nodes)):
        if j != i:
            term = 1 / (nodes[i] - nodes[j])
            for m in range(len(nodes)):
                if m != i and m != j:
                    term *= (x - nodes[m]) / (nodes[i] - nodes[m])

            dLdx_i += term
    return dLdx_i


@numba.njit
def gauss_lobatto_points_weights(n):
    """
    Returns the Gauss-Lobatto points and weights on the interval [0, 1].

    Parameters:
        n (int): The degree of the polynomial (number of points = n + 1).

    Returns:
        points (np.array): Gauss-Lobatto points on [0, 1].
        weights (np.array): Associated weights.
    """
    # Compute the points
    # Transform Chebyshev points to [0, 1] and reverse their order
    # Chebyshev points are cos(pi * k / n) for k = 0, ..., n
    # The transformation is x = 0.5 * (1 - cos(pi * k / n))
    points = np.zeros(n + 1)
    for k in range(n + 1):
        points[k] = 0.5 * (1 - np.cos(np.pi * k / n))

    # Compute the weights
    weights = np.zeros(n + 1)
    weights[0] = weights[n] = 0.5 / n  # Weights at the endpoints
    for i in range(1, n):
        weights[i] = 1 / (n * (n + 1))  # Weights for intermediate points

    return points, weights


@numba.njit
def lagrange_nodal_weights(K, ix, iy):
    """
    Computes the nodal weights using the Lagrange basis, considering a point
    located at (ix,iy) inside the reference spectral element. (ix,iy) \in [0,1].
    Parameters:
    - K (int): The polynomial degree considered.
    - ix (np.array): The x-coordinate inside the element.
    - iy (np.array): The y-coordinate inside the element.
    Returns:
    - NodalWeigths (np.array): The Lagrange nodal weights for the 2D SEM solver.
    """
    points, _ = gauss_lobatto_points_weights(K)
    NodalWeights = np.zeros((K+1)**2)
    for k1 in range(K+1):
        for k2 in range(K+1):
            NodalWeights[k2 + (K+1)*k1] = lagrange_basis(ix,k1,points) * lagrange_basis(iy,k2,points)

    return NodalWeights


@numba.njit
def staticbuildMassAcousticSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h, m):
    """
    Constructs the global mass matrix for a 2D spectral element method (SEM) solver.

    This function constructs the mass matrix for a 2D SEM solver by computing the outer product
    of the 1D mass matrices for each polynomial degree and combining them into a 2D mass matrix.

    Parameters:
    - K (int): The polynomial degree considered.
    - points (np.array): Gauss-Lobatto points on [0, 1].
    - weights (np.array): Associated weights for the Gauss-Lobatto quadrature.
    - Ix (int): The number of elements in the x-direction.
    - Iy (int): The number of elements in the y-direction.
    - boundCond (str): The type of boundary condition (only "dirichlet" accepted here).
    - boundNodes (np.array): The indices of the boundary nodes.
    - h (float): The element size.
    - m (np.array): The material properties for the mass matrix.
    Returns:
    - dataM (np.array): The mass matrix for the 2D SEM solver, flattened into a 1D array.
    - colM (np.array): Column indices for the sparse matrix representation.
    - rowM (np.array): Row indices for the sparse matrix representation.
    """
    
    Mc1D  = np.zeros((K+1,K+1))

    for k in range(K+1):
        for kk in range(K+1):
            for i in range(K+1):
                Mc1D[k,kk] += weights[i] * lagrange_basis(points[i], k, points) * lagrange_basis(points[i], kk, points)

    Mc = np.zeros(((K+1)**2,(K+1)**2)) # 2D mass matrix

    for k1 in range(K+1):
        for k2 in range(K+1):
            k = k1+(K+1)*k2
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    Mc[kp,k] = Mc1D[k1p,k1]*Mc1D[k2p,k2]

    rowM = np.zeros(Ix*Iy*(K+1)**4)
    colM = np.zeros(Ix*Iy*(K+1)**4)
    dataM = np.zeros(Ix*Iy*(K+1)**4)

    nx = int(Ix*K + 1)
    i=0
    for i1 in range(Ix):
        for i2 in range(Iy):
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    for k1 in range(K+1):
                        for k2 in range(K+1):
                            k = k1+(K+1)*k2
                            colM[i] = int(nx*(i2*K+k2)+i1*K+k1)
                            rowM[i] = int(nx*(i2*K+k2p)+i1*K+k1p)

                            dataM[i] = h**2*m[i1,i2]*Mc[kp,k]
                            if boundCond == "dirichlet":
                                if rowM[i] in boundNodes:
                                    if colM[i] == rowM[i]:
                                        dataM[i] = 1
                                    else:
                                        dataM[i] = 0
                            i = i+1

    return dataM, colM, rowM


def buildMassAcousticSEM(domain, model):
    """
    Constructs the global mass matrix for a 2D spectral element method (SEM) solver.

    Accounts for the material properties of the model, such as velocity and density.

    Parameters:
    - domain: Object containing domain properties (e.g., element size, boundary conditions, and node mappings).
    - model: Object containing material properties (e.g., velocity, density).

    Returns:
    - M: Sparse global mass matrix.
    """
      
    K = domain.K

    points, weights = gauss_lobatto_points_weights(K)
    
    Ix = domain.Ix
    Iy = domain.Iy

    h = domain.h
    m = model
    boundCond = domain.boundary_conditions
    boundNodes = domain.boundaryNodes
    boundNodes = np.array(boundNodes, dtype=np.int64)  # Ensure it's a NumPy array to avoid Numba issues

    dataM, colM, rowM = staticbuildMassAcousticSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h, m)
    M = sp.csc_matrix((dataM,(rowM,colM)),shape=((Ix*K+1)*(Iy*K+1),(Ix*K+1)*(Iy*K+1)), dtype=np.float64)

    return M


@numba.njit
def staticbuildStiffnessAcousticSEM(K, points, weights, Ix, Iy, boundCond, boundNodes):
    """
    Constructs the global stiffness matrix for a 2D spectral element method (SEM) solver.

    This function constructs the stiffness matrix for a 2D SEM solver by computing the outer product
    of the 1D stiffness and mass matrices for each polynomial degree and combining them into a 2D stiffness matrix.
    The resulting stiffness matrix is then flattened into a 1D array for efficient storage and computation.

    Parameters:
    - K (int): The polynomial degree considered.
    - points (np.array): Gauss-Lobatto points on [0, 1].
    - weights (np.array): Associated weights for the Gauss-Lobatto quadrature.
    - Ix (int): The number of elements in the x-direction.
    - Iy (int): The number of elements in the y-direction.
    - boundCond (str): The type of boundary condition (only "dirichlet" accepted here).
    - boundNodes (np.array): The indices of the boundary nodes.
    Returns:
    - dataK (np.array): The stiffness matrix for the 2D SEM solver, flattened into a 1D array.
    - rowM (np.array): Row indices for the sparse matrix representation.
    - colM (np.array): Column indices for the sparse matrix representation.
    """
    Mc1D  = np.zeros((K+1,K+1))
    Kc1D  = np.zeros((K+1,K+1))

    for k in range(K+1):
        for kk in range(K+1):
            for i in range(K+1):
                Mc1D[k,kk] += weights[i] * lagrange_basis(points[i], k, points) * lagrange_basis(points[i], kk, points)
                Kc1D[k,kk] += weights[i] * derivative_lagrange_basis(points[i], k, points) * derivative_lagrange_basis(points[i], kk, points)

    Kcy = np.zeros(((K+1)**2,(K+1)**2)) # 2D stiffness matrix in the y-direction
    Kcx = np.zeros(((K+1)**2,(K+1)**2)) # 2D stiffness matrix in the x-direction
    Kc = np.zeros(((K+1)**2,(K+1)**2)) # 2D stiffness matrix

    for k1 in range(K+1):
        for k2 in range(K+1):
            k = k1+(K+1)*k2
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    Kcx[kp,k] = Kc1D[k1p,k1]*Mc1D[k2p,k2]
                    Kcy[kp,k] = Kc1D[k2p,k2]*Mc1D[k1p,k1]

    Kc = Kcx+Kcy

    rowM = np.zeros(Ix*Iy*(K+1)**4)
    colM = np.zeros(Ix*Iy*(K+1)**4)
    dataK = np.zeros(Ix*Iy*(K+1)**4)

    nx = int(Ix*K + 1)
    i=0
    for i1 in range(Ix):
        for i2 in range(Iy):
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    for k1 in range(K+1):
                        for k2 in range(K+1):
                            k = k1+(K+1)*k2
                            colM[i] = int(nx*(i2*K+k2)+i1*K+k1)
                            rowM[i] = int(nx*(i2*K+k2p)+i1*K+k1p)

                            dataK[i] = Kc[kp,k]
                            if boundCond == "dirichlet":
                                if rowM[i] in boundNodes:
                                    if colM[i] == rowM[i]:
                                        dataK[i] = 1
                                    else:
                                        dataK[i] = 0
                            i = i+1

    return dataK, rowM, colM


def buildStiffnessAcousticSEM(domain):
    """
    Constructs the global stiffness matrix for a 2D spectral element method (SEM) acoustic solver.

    Notice that in the acoustic case, the stiffness matrix does not depend on the physical model.
    
    Parameters:
    - domain: Object containing domain properties (e.g., element size, boundary conditions, and node mappings).

    Returns:
    - R: Sparse global stiffness matrix.
    """  
    K = domain.K

    points, weights = gauss_lobatto_points_weights(K)

    Ix = domain.Ix
    Iy = domain.Iy

    boundCond = domain.boundary_conditions
    boundNodes = domain.boundaryNodes
    boundNodes = np.array(boundNodes, dtype=np.int64)  # Ensure it's a NumPy array to avoid Numba issues

    dataK, rowM, colM = staticbuildStiffnessAcousticSEM(K, points, weights, Ix, Iy, boundCond, boundNodes)
    R = sp.csc_matrix((dataK,(rowM,colM)),shape=((Ix*K+1)*(Iy*K+1),(Ix*K+1)*(Iy*K+1)), dtype=np.float64)

    return R


@numba.njit
def staticbuildMassTermSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h):
    """ 
    Constructs the global mass matrix term for a 2D spectral element method (SEM) solver.

    This function constructs the mass matrix term for a 2D SEM solver by computing the outer product
    of the 1D mass matrices for each polynomial degree and combining them into a 2D mass matrix.
    The resulting mass matrix is then flattened into a 1D array for efficient storage and computation.

    Parameters:
    - K (int): The polynomial degree considered.
    - points (np.array): Gauss-Lobatto points on [0, 1].
    - weights (np.array): Associated weights for the Gauss-Lobatto quadrature.
    - Ix (int): The number of elements in the x-direction.
    - Iy (int): The number of elements in the y-direction.
    - boundCond (str): The type of boundary condition (only "dirichlet" accepted here).
    - boundNodes (np.array): The indices of the boundary nodes.
    - h (float): The element size.
    Returns:
    - dataM (np.array): The mass matrix term for the 2D SEM solver, flattened into a 1D array.
    """
    Mc1D  = np.zeros((K+1,K+1))

    for k in range(K+1):
       for i in range(K+1):
            Mc1D[k,k] += weights[i] * lagrange_basis(points[i], k, points) * lagrange_basis(points[i], k, points)

    Mc = np.zeros(((K+1)**2,(K+1)**2)) # 2D mass matrix

    for k1 in range(K+1):
        for k2 in range(K+1):
            k =  k1+(K+1)*k2
            Mc[k,k] = Mc1D[k1,k1]*Mc1D[k2,k2]

    dataM = np.zeros((Ix*K+1)*(Iy*K+1))

    nx = int(Ix*K + 1)
    for i1 in range(Ix):
        for i2 in range(Iy):
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    kg = int(nx*(i2*K+k2p)+i1*K+k1p)

                    dataM[kg] += h**2*Mc[kp,kp]

                    if boundCond == "dirichlet":
                        if kg in boundNodes:
                            dataM[kg] = 1
                        else:
                            dataM[kg] = 0

    return dataM


def buildMassTermSEM(domain):
    """
    Constructs the global mass matrix term for a 2D spectral element method (SEM) solver.

    Parameters:
    - domain: Object containing domain properties (e.g., element size, boundary conditions, and node mappings).

    Returns:
    - M: Sparse global mass term matrix.
    """    
    K = domain.K

    points, weights = gauss_lobatto_points_weights(K)

    Ix = domain.Ix
    Iy = domain.Iy

    h = domain.h
    boundCond = domain.boundary_conditions
    boundNodes = domain.boundaryNodes
    boundNodes = np.array(boundNodes, dtype=np.int64)  # Ensure it's a NumPy array to avoid Numba issues

    dataM = staticbuildMassTermSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h)

    return dataM


@numba.njit
def staticbuildDampingSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h, m):
    """
    Constructs the global damping matrix for a 2D spectral element method (SEM) solver.

    This function constructs the damping matrix for a 2D SEM solver considering absorbing boundary conditions (abc).
    The resulting damping matrix is then flattened into a 1D array for efficient storage and computation.
    Parameters:
    - K (int): The polynomial degree considered.
    - points (np.array): Gauss-Lobatto points on [0, 1].
    - weights (np.array): Associated weights for the Gauss-Lobatto quadrature.
    - Ix (int): The number of elements in the x-direction.
    - Iy (int): The number of elements in the y-direction.
    - boundCond (str): The type of boundary condition (only "abc" accepted here).
    - boundNodes (np.array): The indices of the boundary nodes.
    - h (float): The element size.
    - m (np.array): The material properties for the damping term.
    Returns:
    - dataS (np.array): The damping matrix term for the 2D SEM solver, flattened into a 1D array.
    - rowM (np.array): Row indices for the sparse matrix representation.
    - colM (np.array): Column indices for the sparse matrix representation.
    """
    Mc1D  = np.zeros((K+1,K+1))

    for k in range(K+1):
        for kk in range(K+1):
            for i in range(K+1):
                Mc1D[k,kk] += weights[i] * lagrange_basis(points[i], k, points) * lagrange_basis(points[i], kk, points)

    Sc = np.zeros(((K+1)**2,(K+1)**2))

    for k1 in range(K+1):
        for k2 in range(K+1):
            k = k1+(K+1)*k2
            for k1p in range(K+1):
                for k2p in range(K+1):
                    if boundCond == "abc":
                        kp = k1p+(K+1)*k2p
                        Sc[kp,k] = Mc1D[k1p,k1]*delta(k2p,k2)

    rowM = np.zeros(Ix*Iy*(K+1)**4)
    colM = np.zeros(Ix*Iy*(K+1)**4)
    dataS = np.zeros(Ix*Iy*(K+1)**4)

    nx = int(Ix*K + 1)
    i=0
    for i1 in range(Ix):
        for i2 in range(Iy):
            for k1p in range(K+1):
                for k2p in range(K+1):
                    kp = k1p+(K+1)*k2p
                    for k1 in range(K+1):
                        for k2 in range(K+1):
                            k = k1+(K+1)*k2
                            colM[i] = int(nx*(i2*K+k2)+i1*K+k1)
                            rowM[i] = int(nx*(i2*K+k2p)+i1*K+k1p)
                            if boundCond == "abc":
                                if rowM[i] in boundNodes:
                                    dataS[i] = h*Sc[kp,k]*m[i1,i2]

                            i = i+1

    return dataS, rowM, colM


def buildDampingSEM(domain, model):
    """
    Constructs the global damping matrix for a 2D spectral element method (SEM) solver.

    Parameters:
    - domain: Object containing domain properties (e.g., boundary conditions and node mappings).
    - model: Object containing material properties.

    Returns:
    - S: Sparse global damping matrix.
    """
    K = domain.K
    boundCond = domain.boundary_conditions

    points, weights = gauss_lobatto_points_weights(K)
    
    boundNodes = domain.boundaryNodes
    # Remove upper nodes if the domain has a free surface condition
    if domain.freesurface:
        boundNodes = boundNodes[~np.isin(boundNodes,domain.upperNodes)]
    boundNodes = np.array(boundNodes, dtype=np.int64)  # Ensure it's a NumPy array to avoid Numba issues

    Ix = domain.Ix
    Iy = domain.Iy

    h = domain.h
    m = model

    dataS, rowM, colM = staticbuildDampingSEM(K, points, weights, Ix, Iy, boundCond, boundNodes, h, m)

    S = sp.csc_matrix((dataS,(rowM,colM)),shape=((Ix*K+1)*(Iy*K+1),(Ix*K+1)*(Iy*K+1)), dtype=np.float64)

    return S


@numba.njit
def computeGradient(grad, Pn2, Pn1, Pn0, dt, Mterm, Padj, Ix, Iy, K):
    """
    Computes the gradient in the context of a FWI (Full Waveform Inversion) or inversion problem.
    
    This function computes the gradient by combining the adjoint wavefield and the forward wavefield.
    It reads the forward wavefield from a file, calculates the second derivative of the adjoint wavefield,
    and updates the gradient using the mass matrix.

    Parameters:
    - grad (np.array): The gradient to be updated.
    - Pn2 (np.array): The adjoint wavefield at the current time step.
    - Pn1 (np.array): The adjoint wavefield at the previous time step.
    - Pn0 (np.array): The adjointwavefield at the time step before the previous one.
    - dt (float): The time step size.
    - Mterm (np.array): The mass matrix term.
    - Padj (np.array): The forward wavefield at the current time step.
    - Ix (int): The number of elements in the x-direction.
    - Iy (int): The number of elements in the y-direction.
    - K (int): The polynomial degree of the spectral elements.
    Returns:
    - grad: The calculated gradient.
    """        
    nx = int(Ix*K + 1)

    # 2nd derivative of adjoint wavefield multiplied by the forward wavefield and the mass matrix term.
    for e1 in range(Ix): 
        for e2 in range(Iy):
            for k1 in range(K+1):
                for k2 in range(K+1):
                    e = int(e1+Ix*e2) # ele2ele
                    n = int(nx*(e2*K+k2)+e1*K+k1) # loc2glob
                    grad[e] += (Pn2[n] - 2*Pn1[n] + Pn0[n]) / (dt**2) * Padj[n] * Mterm[n] 

    return grad