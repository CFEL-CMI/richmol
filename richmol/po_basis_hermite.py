import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import factorial,eval_hermite
from matplotlib import pyplot as plt

class PObas:
    def __init__(self):
        pass

    def print_coeffs(self,vec,n):
        print(' '.join(["  %8.4f"%item for item in vec[:,n]]))

    def gen_po_bas(self, molec, ref_coords, POgrid, icoord): #molec, ref_coords,
        """Generate potential-optimized basis set from primitive basis"""

        print("========= Potential-Optimized basis functions ========="+'\n')

        NPOpts = np.size(POgrid)
        print("*** Size of output grid = " + str(NPOpts))
        print("*** Size of output PO basis = " + str(self.NPObas)+'\n')
        #print("*** Output grid for PO functions:"+'\n')
        #print(POgrid)

        #define the output collocation matrix
        bmat = np.zeros((NPOpts,self.NPObas), dtype = float)
        #print(bmat)

        #define quadrature rule
        q, w = self.gen_quad()

        print("*** Constructing 5-point stencil to get harmonic frequency of the PES ***")
        dh = 1e-4 #step for calculating second derivative of PES. [1e-3:1e-9] range is acceptable.
        # five-point stencil to get harmonic frequency of the PES
        stencil_steps = np.array([-2.0*dh, -1.0*dh, 0.0, 1.0*dh, 2.0*dh], dtype=np.float64)
        stencil_coeffs = np.array([-1.0,16.0,-30.0,16.0,-1.0], dtype=np.float64)/(12.0*dh*dh)
        stencil_grid = np.array(np.broadcast_to(ref_coords, (len(stencil_steps),len(ref_coords))))
        stencil_grid[:,self.icoord] = [ref_coords[self.icoord]+dr for dr in  stencil_steps]
        print("Stencil grid: " + str(stencil_grid))
        pes_eq = molec.V(stencil_grid)
        secderiv = np.dot(pes_eq, stencil_coeffs)
        omega = secderiv# second derivative of PES at reference geometry = mu * omega **2

        print("*** Constructing G-matrix at equlilibrium geometry ***")
        gmat = molec.G(np.array([ref_coords]))[0,icoord,icoord] #G(r1,r1,gamma) = G(0,0,0) = mu^-1
        print("Harmonic frequency = " +  str(np.sqrt(secderiv/gmat)))

        print("*** Constructing scaled coordinates ***")
        alpha= np.sqrt(np.sqrt(2.0*np.abs(omega)/np.abs(gmat)))
        r = q / alpha + ref_coords[icoord]
        coords = np.array(np.broadcast_to(ref_coords, (len(r),len(ref_coords))))
        coords[:,icoord] = r[:]

        print("*** Constructing the PES matrix ***")
        Vmat = np.zeros((self.Nbas,self.Nbas), dtype=float)
        pes = molec.V(coords) #PES takes array [r1,r2,gamma] in [angstrom,angstrom,radians] and returns energy in inverse centrimeters

        #print("***  Plot of PES on quadrature grid ***")
        #plt.plot(r,pes[:])
        #plt.show()

        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                fpot = w[:] *  self.hofunc(ni,q[:]) * self.hofunc(nf,q[:]) * pes[:]
                Vmat[ni,nf] = np.sum(fpot)
                #Vmat[nf,ni] = Vmat[ni,nf] #for test
       # print("Potential matrix")
       # print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Vmat]))


        print("*** Constructing the KEO matrix ***")
        Kmat = np.zeros((self.Nbas,self.Nbas), dtype=float)

        print("*** Constructing the G matrix ***")
        gmat = molec.G(coords)[:,icoord,icoord]

        for ni in range(self.Nbas):
            for nf in range(ni,self.Nbas):
                fkeo = -0.5 * w[:] *  self.hofunc(ni,q[:]) * self.ddhofunc(nf,q[:]) * gmat[:]
                Kmat[ni,nf] = np.sum(fkeo)  * alpha**2
                #Kmat[nf,ni] = Kmat[ni,nf] # for test
        #print("KEO matrix")
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Kmat]))


        print("Operators on 1D quadrature grid\n" + "%23s"%"q" + "%18s"%"w" \
                + "%18s"%"r" + "%18s"%"KEO" + "%18s"%"PES")
        for i in range(len(r)):
            print(" %4i"%i + "  %16.8f"%q[i] + "  %16.8e"%w[i] \
            + "  %16.8f"%r[i] + "  %16.8f"%gmat[i] + "  %16.8f"%pes[i])


        print("*** Constructing the Hamiltonian matrix ***")
        Hmat = Vmat + Kmat
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
                #Smat[nf,ni] = Smat[ni,nf]

        #print("Overlap matrix")
        #print('\n'.join([' '.join(["  %15.8f"%item for item in row]) for row in Smat]))

        print("*** Constructing the collocation matrix in the PO basis ***")
        for n in range(self.NPObas):
            for k in range(NPOpts):
                for iho in range(self.Nbas):
                    bmat[k,n]  += vec[iho,n] * self.hofunc(iho,POgrid[k]) * np.exp(-0.5 * POgrid[k]**2)

        #print coefficients
        #self.print_coeffs(vec,2)

        #plot wavefunctions
        for n in range(2):
            plt.plot(POgrid,self.hofunc(n,POgrid[:]) * np.exp(-0.5 * POgrid[:]**2))
            plt.plot(POgrid,-bmat[:,n])
        plt.show()
        return bmat

class HObas(PObas):
    """ One-dimensional basis of Harmonic Oscillator eigenfunctions"""
    """ Inherits after PObas class, with which it constructs the PO basis expressed in the Harmonic Oscillator Basis"""
    """ For derivation details ask see Richmol theory manual or ask Emil Zak"""

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
        self.ref_coords  = ref_coords
        self.icoord = 0

    def gen_quad(self):
        # Gauss-Hermite quadrature of degree Nquad
        x, w = hermgauss(self.Nquad)
        return x,w

    def hofunc(self,n,q):
        """Return the value of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms). Note: weight function has been removed"""
        if n < 0:
            return 0
        else:
            # Normalization constant for state n
            norm = lambda n: 1./np.sqrt(np.sqrt(np.pi)*2**n*factorial(n))
            return norm(n) * eval_hermite(n,q)

    def dhofunc(self,n,q):
        """Return the value of the first derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        return (np.sqrt(float(n)) * self.hofunc(n-1,q) + np.sqrt(float(n)+1.)* self.hofunc(n+1,q))/np.sqrt(2.)

    def ddhofunc(self,n,q):
        """Return the value of the second derivatibe of the n-th Harmonic Oscillator eigenfunction at grid point q (angstroms)."""
        return (q**2 - (2*n+1)) * self.hofunc(n,q)

    def quad_weight_func(self,q):
        """Return inverse of the Gauss-Hermite weight function"""
        return np.exp( q * q )



def get_turning_points(n):
    """Return the classical turning points for the HO state n."""
    qmax = np.sqrt(2. * (float(n) + 0.5)) #dimensionless units
    return qmax


if __name__=="__main__":

    from mol_xy2 import XY2_ralpha
    import poten_h2s_Tyuterev

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

    # generate grid for PO functions
    print("Turning point of the Nbas Harmonic Oscillator function = " + str(get_turning_points(Nbas) ))
    POgrid = np.linspace(-2.0 * get_turning_points(Nbas) , 2.0 * get_turning_points(Nbas) , NPO_grid)

    strBas = HObas(Nbas,Nquad,NPObas,POgrid,ref_coords)
    bmat  = strBas.gen_po_bas(h2s,ref_coords,POgrid,0)

    # reference Numerov stretching energies for H2S from TROVE
    trove_str_enr = [0.00000000, 2631.91316250, 5168.60694305, 7610.21707943, 9956.62949349, \
                     12207.53640553,14362.48285216,16420.90544929,18382.16446603,20245.56943697, \
                     22010.39748652,23675.90513652,25241.42657687,26708.10237789,28092.75163810, \
                     29462.97307758,30918.23065498,32507.23053835,34229.22373259,36072.32685441, \
                     38026.38406259]

    print("richmol-TROVE for stretching")
    print(" ".join("  %12.4f"%e1 + "  %12.4f"%e2 + "  %12.4f"%(e1-e2) + "\n" for e1,e2 in zip(strBas.energy, trove_str_enr)))
