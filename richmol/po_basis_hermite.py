import numpy as np
#from molecule import G as Gmat
#from molecule import V as PES
import quadpy

def po_hermite( Nbas, Nquad, NPO_pts, NPO_bas):
    """Calculate collocation matrix for potential-optimized (PO) 1D basis functions associated with stretching coordinates (Hermite)

     Args:
            Mquad (int): number of quadrature points in Gauss-Hermite integration over the primitive basis (Harmonic Oscillator)
            Nbas (int): number of primitive basis functions used to represent PO basis functions
            NPO_pts (int): size of grid onto which the PO basis functions are stored
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
    print("========= Primitive basis type: Harmonic Oscillator basis =========")
    print("========= Quadrature rule: Gauss-Hermite =========")

    bmat = np.zeros((NPO_pts,NPO_bas), dtype = float)
    print(bmat)


    scheme = quadpy.e1r2.gauss_hermite(5)
    scheme.show()
    val = scheme.integrate(lambda x: x ** 2)

    return bmat




po_hermite(1, 1, 10, 10)