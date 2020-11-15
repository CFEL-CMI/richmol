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

    def gen_po_bas(self, molec, ref_coords,POgrid, icoord): #molec, ref_coords, 
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
  

        gmat = molec.G(np.array([ref_coords]))[0,icoord,icoord] #G(r1,r1,gamma) = G(0,0,0) = mu^-1

        #move to function
        dh = 1e-4 #step for calculating second derivative of PES. [1e-3:1e-9] range is acceptable.
        # five-point stencil to get harmonic frequency of the PES
        stencil_steps = np.array([-2.0*dh, -1.0*dh, 0.0, 1.0*dh, 2.0*dh], dtype=np.float64)
        stencil_coeffs = np.array([-1.0,16.0,-30.0,16.0,-1.0], dtype=np.float64)/(12.0*dh*dh)
        stencil_grid = np.array(np.broadcast_to(ref_coords, (len(stencil_steps),len(ref_coords))))
        stencil_grid[:,icoord] = [ref_coords[icoord]+dr for dr in  stencil_steps]
        print("Stencil grid" + str(stencil_grid))
        pes_eq = molec.V(stencil_grid)
        secderiv = np.dot(pes_eq, stencil_coeffs)
        omega = secderiv#np.sqrt(secderiv * gmat) # second derivative of PES at reference geometry = mu * omega **2
        print("Harmonic frequency = " +  str(np.sqrt(secderiv/gmat)))


        #Pull the values of omega and mu constants from second derivative of the PES and the G-matrix
        #gmat = self.gen_gmat()
        # mapping between r and x


       # plt.plot(r,pes[:])
        #plt.show()
        qscale = np.sqrt(np.sqrt( 2.0*np.abs(omega)/np.abs(gmat) ))
        #qscale = np.sqrt(2.0 * omega / gmat)
        r = q / qscale + ref_coords[icoord]
     

        coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
        coords[:,icoord] = r[:]
        gmat = molec.G(coords)[:,icoord,icoord]
        print(f"Mapping q <--> r calculated for Gauss-Hermite quadrature: {qscale}, (mu={gmat}, omega={omega})")


        print("*** Integrating the PES part ***")
        coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
        #print(coords)
        coords[:,0] = r[:]
        #print(coords)
        pes = molec.V(coords) #PES takes array [r1,r2,gamma] in [angstrom,angstrom,radians] and returns energy in inverse centrimeters
        #print(pes)
        #print(np.shape(pes))
        #pes = lambda x: 0.5 * x ** 2 + 0.05 * x ** 4 #analytic test PES 
    

  
        Vmat = np.zeros((self.Nbas,self.Nbas), dtype=float)

        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                fpot = w[:] *  self.hofunc(ni,q[:]) * self.hofunc(nf,q[:]) * pes[:] 
                Vmat[ni,nf] = np.sum(fpot) 
                    #Vmat[ni,nf] +=  w[k] *  self.hofunc(ni,q[k]) * self.hofunc(nf,q[k]) * pes[k] *qscale *qscale
                    #Vmat[nf,ni] = Vmat[ni,nf] #test
        print("Potential matrix")
       # print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Vmat]))


        print("*** Integrating the KEO part ***")
        Kmat = np.zeros((self.Nbas,self.Nbas), dtype=float)
        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                fkeo = -0.5 * w[:] *  self.hofunc(ni,q[:]) * self.ddhofunc(nf,q[:]) * gmat[:] 
                Kmat[ni,nf] = np.sum(fkeo)  * qscale**2
                    #Kmat[nf,ni] = Kmat[ni,nf] # for test
        print("KEO matrix")
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Kmat]))


        print("Operators on 1D quadrature grid\n" + "%23s"%"x" + "%18s"%"w" \
                + "%18s"%"r" + "%18s"%"keo" + "%18s"%"poten")
        for i in range(len(r)):
            print(" %4i"%i + "  %16.8f"%q[i] + "  %16.8e"%w[i] \
            + "  %16.8f"%r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%pes[i])

        Hmat = Vmat + Kmat
        print("*** Hamiltonian matrix ***")
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Hmat]))

        print("*** Diagonalizing the 1D Hamiltonian ***")

        eval, vec = np.linalg.eigh(Hmat,UPLO='U')
        self.energy = eval - eval[0]

        #verify orthonormality of basis
        Smat = np.zeros((self.Nbas,self.Nbas), dtype=float)
        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                foverlap =  w[:] *  self.hofunc(ni,q[:]) * self.hofunc(nf,q[:])
                Smat[ni,nf] = np.sum(foverlap) 
                    #Smat[nf,ni] = Smat[ni,nf] #test
        print("Overlap matrix")
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Smat]))
        
        print(np.dot(vec[:,1],vec[:,2]))
        print(np.dot(vec[:,1],vec[:,1]))

        print("*** Constructing collocation matrix in PO basis ***")

        for n in range(self.NPObas):
            for k in range(NPOpts):
                for iho in range(self.Nbas):
                    bmat[k,n]  += vec[iho,n] * self.hofunc(iho,POgrid[k])

       
        #plt.show()
        #print(bmat)
        #for n in range(3):
        #    plt.plot(POgrid,bmat[:,n]**2)
        #plt.show()
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
        dh = 1e-4 #step for calculating second derivative of PES. [1e-3:1e-9] range is acceptable.
        # five-point stencil to get harmonic frequency of the PES
        stencil_steps = np.array([-2.0*dh, -1.0*dh, 0.0, 1.0*dh, 2.0*dh], dtype=np.float64)
        stencil_coeffs = np.array([-1.0,16.0,-30.0,16.0,-1.0], dtype=np.float64)/(12.0*dh*dh)
        stencil_grid = np.array(np.broadcast_to(ref_coords, (len(stencil_steps),len(ref_coords))))
        stencil_grid[:,icoord] = [ref_coords[icoord]+dr for dr in  stencil_steps]
        print("Stencil grid" + str(stencil_grid))
        pes_eq = molec.V(stencil_grid)
        omega = np.dot(pes_eq, stencil_coeffs)
        print("Harmonic frequency = " +  str(omega))
        return omega

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

    def hofunc(self,n,q):
        """Return the value of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        # Normalization constant for state n
        if n < 0:
            return 0
        else:
            norm = lambda n: 1./np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))
            return norm(n) * eval_hermite(n,q) 

    def dhofunc(self,n,q):
        """Return the value of the first derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        # Normalization constant for state n
        return (np.sqrt(float(n)) * self.hofunc(n-1,q) + np.sqrt(float(n)+1.)* self.hofunc(n+1,q))/np.sqrt(2.)

    def ddhofunc(self,n,q):
        """Return the value of the second derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        # Normalization constant for state n
        return (q**2 - (2*n+1)) * self.hofunc(n,q)

    def quad_weight_func(self,q):
        """Return inverse of the Gauss-Hermite weight function"""
        return np.exp( q * q )




def get_turning_points(n):
    """Return the classical turning points for the HO state n."""
    qmax = np.sqrt(2. * (float(n) + 0.5)) #unscaled
    return qmax


if __name__=="__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev
    import poten_h2o_Polyansky

    # equilibrium/reference coordinates
    ref_coords = [1.3359007, 1.3359007, 92.265883/180.0*np.pi]

    # H2S, using valence-bond coordinates and Tyuterev potential
    h2s = XY2_ralpha(masses=[31.97207070, 1.00782505, 1.00782505], poten=poten_h2s_Tyuterev.poten)

    # test KEO and potential
    G = h2s.G(np.array([ref_coords]))
    V = h2s.V(np.array([ref_coords]))

    NPO_grid = 200
    Nbas = 100
    Nquad = 100
    NPObas = 5


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

    bmat  = strBas.gen_po_bas(h2s,ref_coords,POgrid,0)

    #for n in range(3):
    #    plt.plot(POgrid,strBas.hofunc(n,POgrid,alpha)**2)
    #plt.show()


    print("Stretching energy levels")
    print(" ".join("  %12.4f"%energy  + "\n" for energy in strBas.energy))