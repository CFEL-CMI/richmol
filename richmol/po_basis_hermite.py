import numpy as np
#from molecule import G as Gmat
#from molecule import V as PES
import quadpy
from matplotlib import pyplot as plt
from scipy.special import factorial
from scipy.constants import hbar

def make_Hr(Nbas):
    """Return a list of np.poly1d objects representing Hermite polynomials."""

    # Define the Hermite polynomials up to order Nbas by recursion:
    # H_[n] = 2qH_[n-1] - 2(n-1)H_[n-2]
    Hr = [None] * (Nbas + 1)
    Hr[0] = np.poly1d([1.,])
    Hr[1] = np.poly1d([2., 0.])
    for n in range(2, Nbas+1):
        Hr[n] = Hr[1]*Hr[n-1] - 2*(n-1)*Hr[n-2]
    return Hr

def get_turning_points(n):
    """Return the classical turning points for the HO state n."""
    qmax = np.sqrt(2. * (float(n) + 0.5)) #unscaled
    return qmax

def hofunc(n,q,alpha,Hr):
    """Return a value of the n-th Harmonic Oscillator eigenfunction at grid point q."""
    #alpha(float): inverse natural length unit
    # Normalization constant and energy for vibrational state v
    norm = lambda n: 1./np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))

    return norm(n) * Hr[n](alpha * q) * np.exp(- alpha * alpha * q * q / 2.)

def ddhofunc(n,q,alpha,Hr):
    """Return a value of second derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q."""
    #alpha(float): inverse natural length unit
    # Normalization constant and energy for vibrational state v
    norm = lambda n: 1./np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))

    return (alpha**2 * q**2 - alpha*(2*n+1)) * norm(n) * Hr[n](alpha * q) * np.exp(- alpha * alpha * q * q / 2.)


def quad_weight_func(q,alpha):
    """Return inverse of the Gauss-Hermite weight function"""
    return np.exp( alpha * alpha * q * q )

def Gdiag(q):
     """Return a value of the G-matrix for coordinate 1 at grid point q."""
     return -0.5

def po_hermite( Nbas, Nquad, PO_grid, NPO_bas):
    """Calculate collocation matrix for potential-optimized (PO) 1D basis functions associated with stretching coordinates (Hermite)

     Args:
            Mquad (int): number of quadrature points in Gauss-Hermite integration over the primitive basis (Harmonic Oscillator)
            Nbas (int): number of primitive basis functions used to represent PO basis functions
            PO_grid (array(NPO_pts)):  grid array onto which the PO basis functions are stored
            NPO_bas (int): size of PO basis

    Returns:
            bmat (array (NPO_pts, NPO_bas): collocation matrix of values of PO basis functions on a grid

    Requires:
          1) G-matrix routine
          2) Potential energy surface (PES) routine
          3) second derivative of PES
        """
    
    print("========= Potential-Optimized basis functions =========")
    print(" ")
    print("========= Coordinate type:  Stretching  =========")
    print("========= Primitive basis type: Harmonic Oscillator basis of size " +str(Nbas) +" =========")
    print("========= Quadrature rule: Gauss-Hermite of size " +str(Nquad) +"=========")

    NPO_pts = np.size(PO_grid)

    print("*** Size of output PO basis = " + str(NPO_bas))
    print("*** Output grid for PO functions:")
    print(PO_grid)

    print("*** Size of output grid = " + str(NPO_pts))
    
    #Initialize the output collocation matrix
    bmat = np.zeros((NPO_pts,NPO_bas), dtype = float)
    #print(bmat)

    #construct the object of Hermite polynomials
    Hr = make_Hr(Nbas)

    #Pull the values of omega and mu constants from second derivative of the PES and the G-matrix
    omega = 1.0
    mu = 1.0
    # Scaling constant for length = sqrt(mu * omega / hbar)
    alpha = np.sqrt(mu * omega )


    print("*** Integrating the PES part ***")
    PES_temp = lambda x: 0.5 * x ** 2 + 1/6. * x ** 4 #temporary PES - needs to be replaced by the real PES from molecule.py
    
    #PES on the grid
    #plt.plot(x,PES_temp(x))
    #plt.show()
    
    #Example HO basis function
    #print(hofunc(1,x,alpha,Hr))
    plt.plot(PO_grid,hofunc(2,PO_grid,alpha,Hr))
    plt.show()

    scheme = quadpy.e1r2.gauss_hermite(Nquad)
    #scheme.show()

    Vmat = np.zeros((Nbas,Nbas), dtype=float)


    for ni in range(Nbas):
        for nf in range(ni,Nbas):
            Vmat[ni,nf] = scheme.integrate(lambda q:  quad_weight_func(q,alpha) * hofunc(ni,q,alpha,Hr) * hofunc(nf,q,alpha,Hr) * PES_temp(q))
            Vmat[nf,ni] = Vmat[ni,nf] #test
    
    #print('\n'.join([''.join(['{:10.5}'.format(item) for item in row]) for row in Vmat]))


    print("*** Integrating the KEO part ***")
    Kmat = np.zeros((Nbas,Nbas), dtype=float)
    for ni in range(Nbas):
        for nf in range(ni,Nbas):
            Kmat[ni,nf] = scheme.integrate(lambda q:  quad_weight_func(q,alpha) * hofunc(ni,q,alpha,Hr) * ddhofunc(nf,q,alpha,Hr) * Gdiag(q))
            Kmat[nf,ni] = Kmat[ni,nf] # for test
    

    Hmat = Vmat + Kmat
    print('\n'.join([''.join(['{:10.5}'.format(item) for item in row]) for row in Hmat]))
    print("*** Diagonalizing the 1D Hamiltonian ***")

    eval, vec = np.linalg.eigh(Hmat,UPLO='U')

    #verify orthonormality
    #print(np.dot(vec[:,1],vec[:,2]))
    #print(np.dot(vec[:,1],vec[:,1]))

    print("*** Constructing collocation matrix in PO basis ***")

    for n in range(NPO_bas):
        for k in range(NPO_grid):
            for iho in range(Nbas):
                bmat[k,n]  += vec[iho,n] * hofunc(iho,PO_grid[k],alpha,Hr)

    plt.plot(PO_grid,bmat[:,2])
    plt.show()
    #print(bmat)
    return bmat



NPO_grid = 200
Nbas = 10
Nquad = 10
NPO_bas = 10

#Pull the values of omega and mu constants from second derivative of the PES and the G-matrix
omega = 1.0
mu = 1.0 #these are going to be functions

# Scaling constant for length = sqrt(mu * omega / hbar)
alpha = np.sqrt(mu * omega)

# generate grid for PO functions
print("Turning point of Nbas HO function = " + str(get_turning_points(Nbas) * alpha))
PO_grid = np.linspace(-2.0 * get_turning_points(Nbas) * alpha , 2.0 * get_turning_points(Nbas) * alpha, NPO_grid)
#alternatively PO_grid = np.arange(-2.0 * get_turning_points(Nbas) * alpha , 2.0 * get_turning_points(Nbas) * alpha,0.1)
#print(PO_grid)

po_hermite(Nbas, Nquad, PO_grid, NPO_bas)