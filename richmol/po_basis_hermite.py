import numpy as np
from numpy.polynomial.hermite import hermgauss, hermval, hermder
from matplotlib import pyplot as plt
from scipy.special import factorial,eval_hermite
from scipy.constants import hbar


class PObas:
    def __init__(self):
        pass

    def gen_gmat(self):
        return -0.5

    def gen_po_bas(self, POgrid, icoord): #molec, ref_coords, 
        """Generate potential-optimized basis set from primitive basis"""

        print("========= Potential-Optimized basis functions ========="+'\n')

        # get KEO and PES on quadrature grid

        NPOpts = np.size(POgrid)
        print("*** Size of output grid = " + str(NPOpts))
        print("*** Size of output PO basis = " + str(self.NPObas)+'\n')
        print("*** Output grid for PO functions:"+'\n')
        print(POgrid)

        #define the output collocation matrix
        bmat = np.zeros((NPOpts,self.NPObas), dtype = float)
        #print(bmat)

        #define quadrature rule
        q,w = self.gen_quad()
  
        #Pull the values of omega and mu constants from second derivative of the PES and the G-matrix
        gmat = self.gen_gmat()

        print("*** Integrating the PES part ***")
        pes = lambda x: 0.5 * x ** 2 #+ 1/6. * x ** 4 #temporary PES - needs to be replaced by the real PES from molecule.py
    
        #PES on the grid
        #plt.plot(x,pes(x))
        #plt.show()
        alpha = 1.0
        Vmat = np.zeros((self.Nbas,self.Nbas), dtype=float)

        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                for k in range(len(q)):
                    Vmat[ni,nf] +=  w[k] * self.quad_weight_func(q[k],alpha) * self.hofunc(ni,q[k],alpha) * self.hofunc(nf,q[k],alpha) * pes(q[k])
                    Vmat[nf,ni] = Vmat[ni,nf] #test
    
        #print('\n'.join([''.join(['{:10.5}'.format(item) for item in row]) for row in Vmat]))


        print("*** Integrating the KEO part ***")
        Kmat = np.zeros((self.Nbas,self.Nbas), dtype=float)
        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                 for k in range(len(q)):
                    Kmat[ni,nf] += w[k] * self.quad_weight_func(q[k],alpha) * self.hofunc(ni,q[k],alpha) * self.ddhofunc(nf,q[k],alpha) * gmat
                    Kmat[nf,ni] = Kmat[ni,nf] # for test
        

        Hmat = Vmat + Kmat
        print("*** Hamiltonian matrix ***")
        print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Hmat]))

        print("*** Diagonalizing the 1D Hamiltonian ***")

        eval, vec = np.linalg.eigh(Hmat,UPLO='U')
        self.energy = eval - eval[0]
        #verify orthonormality
        #print(np.dot(vec[:,1],vec[:,2]))
        #print(np.dot(vec[:,1],vec[:,1]))

        print("*** Constructing collocation matrix in PO basis ***")

        for n in range(self.NPObas):
            for k in range(NPOpts):
                for iho in range(self.Nbas):
                    bmat[k,n]  += vec[iho,n] * self.hofunc(iho,POgrid[k],alpha)

        #plt.plot(POgrid,bmat[:,2])
        #plt.show()
        #print(bmat)
        return bmat

class HObas(PObas):
    """ One-dimensional basis of Harmonic Oscillator eigenfunctions"""
    """ Inherits after PObas class, with which it constructs the PO basis expressed in the Harmonic Oscillator Basis"""

    """ Note: in coordinates used:
                                     index 0 = r1 stretching
                                     index 1 = r2 stretching
                                     index 2 = gamma bending """
    def __init__(self,Nbas,Nquad,NPObas,POgrid,ref_coords):
        """
        Args:
           
            Nbas (int): number of primitive Harmonic Oscillator basis functions used to represent the PO basis functions
            Nquad (int): number of quadrature points in Gauss-Hermite integration over the primitive basis (Harmonic Oscillator)
            NPObas (int): size of the PO basis
            POgrid (array(NPOpts)): 1D grid array on which the PO basis functions are stored for multi-d integration. Angstroms.

            molec (Molecule): Information about the molecule, KEO and PES.
            ref_coords (array (no_coords=3)): Reference values of all internal coordinates.
     
        Returns:
            bmat (array (NPOpts, NPObas)): collocation matrix for the PO basis functions (values of PO basis functions on the provided grid)

        Requires:
            1) G-matrix 
            2) Potential energy surface (PES) 
            3) second derivative of PES to get natural length units (in angstroms)
        """
        self.Nbas = Nbas
        self.Nquad = Nquad
        self.NPObas = NPObas
        self.POgrid = POgrid
        #self.molec = molec
        self.ref_coords  = ref_coords
        self.icoord = 0
        self.toangstr = 1.0 #conversion from m to angstroms in natual length unit of Harmonic Oscillator

     
    def gen_quad(self):
        # Gauss-Hermite quadrature of degree Nquad
        x, w = hermgauss(self.Nquad)
        return x,w

    def get_omega(self):
    # calculate the characteristic harmonic frequency (omega) for the 1D potential energy surface cut
        # use finite-differences (7-point) to compute frequency
        fdf_steps = np.array([3*fdf_h, 2*fdf_h, fdf_h, 0.0, -fdf_h, -2*fdf_h, -3*fdf_h], dtype=np.float64)
        fdf_coefs = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
        fdf_denom = 180.0
        coords = np.array(np.broadcast_to(ref_coords, (len(fdf_steps),len(ref_coords))))
        coords[:,icoord] = [ref_coords[icoord]+st for st in fdf_steps]
        poten = molec.V(coords)
        freq = np.dot(poten, fdf_coefs)/(fdf_denom*fdf_h*fdf_h)
        return freq

    def get_mu(self):
        # calculate the reduced mass associated with the stretching coordinate
        mu = molec.G(np.array([ref_coords]))[0,0,0]
        return mu

    def calc_alpha(self):
        """ calculate natural length unit for the 1D Harmonic Oscillator basis (in angstroms)"""
        alpha = np.sqrt(self.get_mu  * self.get_omega / hbar) * toangstr
        return alpha

    def map_quad_nodes(self,x):
        # transform the crude quadrature nodes into internal coordinate values
        xmap = np.sqrt(np.sqrt( 2.0*np.abs(self.get_omega)/np.abs(self.get_mu) ))
        q = x / xmap + self.ref_coords[self.icoord]
        return q

    def hofunc(self,n,q,alpha):
        """Return the value of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        # Normalization constant for state n
        norm = lambda n: 1./np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))
        return norm(n) * np.sqrt(alpha) * eval_hermite(n,alpha * q) * np.exp(- alpha * alpha * q * q/ 2.)

    def ddhofunc(self,n,q,alpha):
        """Return the value of the second derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        # Normalization constant for state n
        return (alpha**2 * q**2 - alpha*(2*n+1)) * self.hofunc(n,q,alpha)

    def quad_weight_func(self,q,alpha):
        """Return inverse of the Gauss-Hermite weight function"""
        return np.exp( alpha * alpha * q * q )




def get_turning_points(n):
    """Return the classical turning points for the HO state n."""
    qmax = np.sqrt(2. * (float(n) + 0.5)) #unscaled
    return qmax


if __name__=="__main__":

    NPO_grid = 200
    Nbas = 5
    Nquad = 5
    NPObas = 5
    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    #Pull the values of omega and mu constants from second derivative of the PES and the G-matrix
    omega = 1.0
    mu = 1.0 #these are going to be functions

    # Scaling constant for length = sqrt(mu * omega / hbar)
    alpha = np.sqrt(mu * omega)
    # generate grid for PO functions
    print("Turning point of Nbas HO function = " + str(get_turning_points(Nbas) * alpha))
    POgrid = np.linspace(-2.0 * get_turning_points(Nbas) * alpha , 2.0 * get_turning_points(Nbas) * alpha, NPO_grid)
    #POgrid = np.linspace(-10.0  , 10.0 , NPO_grid)
    #alternatively PO_grid = np.arange(-2.0 * get_turning_points(Nbas) * alpha , 2.0 * get_turning_points(Nbas) * alpha,0.1)
    #print(PO_grid)
    strBas = HObas(Nbas,Nquad,NPObas,POgrid,ref_coords)

    bmat  = strBas.gen_po_bas(POgrid,0)

    #for n in range(Nbas):
    #    plt.plot(POgrid,strBas.hofunc(n,POgrid,alpha))
    #plt.show()


    print("Stretching energy levels")
    print(" ".join("  %12.4f"%energy  + "\n" for energy in strBas.energy))