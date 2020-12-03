import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from mapping import indexmap
from matplotlib import pyplot as plt
class MVP:
    def __init__(self,Nquad1D_herm,Nquad1D_leg):
        self.w_tol = 1e-30 #threshold value for keeping 3D quadrature product-weights
        self.Nquad1D_herm =  Nquad1D_herm  #number of Gauss-Hermite quadrature points in 1D
        self.Nquad1D_leg =  Nquad1D_leg  #number of Gauss-Legendre quadrature points in 1D


    def gen_herm_quad(self):
        # Gauss-Hermite quadrature of degree Nquad
        x, w = hermgauss(self.Nquad1D_herm)
        return x,w

    def gen_leg_quad(self):
        # quadratures
        x, w = leggauss(self.Nquad1D_leg)

        # quadrature abscissas -> coordinate values
        r = np.arccos(x)
        return r,w

    def f_temp(self,r1,r2,theta,i,j):
        """temporary test function for 3D integration"""
        return r1 * r2 * theta * np.exp(-r1**2) * np.exp(-r2**2)

    def helem(self,leftindex,rightindex):


        #  w1, w2, w3 are weights over a pruned grid x1, x2, x3
        for k1 in range(len(w1)):
            for k2 in range(len(w2)):
                for k3 in range(len(w3)):
                    hij += self.f_temp(x1[k1],x2[k2],x3[k3],leftindex,rightindex) * w1[k1] * w1[k2] * w1[k3]
        return hij


    def calc_hmat(self):
        """calculate full Hamiltonian Matrix"""
        
        """construct the quadrature grid. For now it is direct product Gaussian grid"""
        x1, w1 = self.gen_herm_quad()
        x2 = x1
        w2 = w1
        x3, w3 = self.gen_leg_quad()

        """print(x3,w3)
        plt.stem(x3,w3)
        plt.show()"""

        grid_kron = np.kron(x1,x2)
        weights_kron = np.kron(w1,w2)

        #prune the grid by weights
        if weights_kron[:] > self.w_tol:
            weights = weights_kron
        print(weights)

        """calculate the <psi_i | H | psi_j> integral"""
        #self.helem(leftindex,rightindex)


if __name__=="__main__":

    """generate mapping function"""
    b = 4

    simpleMap = indexmap(b,'simple',3)
    indmap =simpleMap.gen_map()
    print(indmap)

    ham0 = MVP(10,10)
    ham0.calc_hmat()