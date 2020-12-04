import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from mapping import indexmap
from matplotlib import pyplot as plt
from test import eval_k


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

    def f_temp(self,ivec,jvec,coords):
        """temporary test function for 3D integration
        ivec: a vector of shape (3, ) containing i1,i2,i3
        jvec : a vector of shape (3, ) containing j1,j2,j3
        coords: (3, ): r1,r2,theta"""
        return coords[0]**2 * coords[1]**2 * coords[2]**2 * np.exp(-coords[0]**2) * np.exp(-coords[1]**2)

    def helem(self,ivec,jvec,coords,weights):
        """Calculated single matrix element using gaussian quadratures"""
        #print(np.shape(coords[:,1]))
        #weightsvec[:] = weights[:,0] * weights[:,1] * weights[:,2]
        #print(np.shape(weightsvec))
        hij = 0
        for ipoint in range(np.size(coords,axis=0)):
            #print(self.f_temp(ivec,jvec,coords[ipoint,:]))
            #print(weights[ipoint,0] * weights[ipoint,1] * weights[ipoint,2])
            #hij += self.f_temp(ivec,jvec,coords[ipoint,:]) * weights[ipoint,0] * weights[ipoint,1] * weights[ipoint,2]
            hij += eval_k(ivec, jvec, ipoint)  * weights[ipoint,0] * weights[ipoint,1] * weights[ipoint,2]
         
        #fvec = self.f_temp(ivec,jvec,coords[:])
        #hij = np.dot( fvec, weights)
        return hij

    def helem_lame(self,ivec,jvec,w1,w2,w3):
        """Calculated single matrix element using gaussian quadratures.
        the integrand function is only defined by grid indices which must be the gaussian quadrature points
        """
        hij = 0

        for k1 in range(self.Nquad1D_herm):
            for k2 in range(self.Nquad1D_herm):
                for k3 in range(self.Nquad1D_leg):
                    #print(eval_k([0,0,0],[1,1,1], [k1,k2,k3])  * w1[k1] *  w2[k2] * w3[k3])
                    #hij += eval_k([indmap[ivec,0],indmap[ivec,1],indmap[ivec,2]], [indmap[jvec,0],indmap[jvec,1],indmap[jvec,2]], [k1,k2,k3])  * w1[k1] *  w2[k2] * w3[k3]
                    hij += eval_k(ivec,jvec, [k1,k2,k3])  * w1[k1] *  w2[k2] * w3[k3]
        return hij



    def calc_hmat(self,indmap):
        """calculate full Hamiltonian Matrix"""
        #indmap: array [Nbasis,4] with mapping between basis set indices and integers
        Nbas = np.size( indmap , axis =0)
        print("number of basis functions = " + str(Nbas))
        """construct the quadrature grid. For now it is direct product Gaussian grid"""
        x1, w1 = self.gen_herm_quad()
        x2 = x1
        w2 = w1
        x3, w3 = self.gen_leg_quad()

        """print(x3,w3)
        plt.stem(x3,w3)
        plt.show()"""

        grid_dp = np.array(np.meshgrid(x3, x2, x1, indexing = 'ij')).T.reshape(-1,3)

        #print(grid_dp)
        #print(np.shape(grid_dp))
        #print(type(grid_dp))

        weights_dp = np.array(np.meshgrid(w3, w2, w1, indexing = 'ij')).T.reshape(-1,3)

        #print(x1[:],x2[:],x3[:])
        #weights_12 = np.tensordot(w1,w2,axes=0)
        #print(np.shape(grid_12))
        #weights_dp = np.tensordot(weights_12,w3,axes=0)
        #print(np.shape(weights_dp))
        #print(weights_dp)
        """counter= 0
        for i in range(0,10):
            for j in range(0,10):            
                for k in range(0,10):
                    print(x3[k]-grid_dp[counter,0],x2[j]-grid_dp[counter,1],x1[i]-grid_dp[counter,2]) #this is the indexing
                    counter +=1"""

        #weights_12 = np.kron(w1,w2)
        #weights_dp = np.kron(weights_12,w3)
        #print(weights_dp)
        #prune the grid by weights
        #for ipoint in range(len(weights_dp)):
        #    if weights_dp[ipoint] > self.w_tol:
        #        weights = weights_dp[ipoint]
        #print(weights)

        hmat = np.zeros((Nbas,Nbas), dtype = float)
        """calculate the <psi_i | H | psi_j> integral """
        for i in range(Nbas):
            ivec = [indmap[i][0],indmap[i][1],indmap[i][2]]
            print(type(ivec))
            print(ivec)
            for j in range(Nbas):
                jvec = [indmap[j][0],indmap[j][1],indmap[j][2]]
                #hmat[ivec,jvec] = self.helem(ivec,jvec,grid_dp,weights_dp)
                hmat[i,j] = self.helem_lame(ivec,jvec,w1,w2,w3)


        print(hmat)
        eval, eigvec = np.linalg.eigh(hmat)
        print(eval)



if __name__=="__main__":

    """generate mapping function"""
    b = 2

    simpleMap = indexmap(b,'simple',3)
    indmap =simpleMap.gen_map()
    #print(indmap)

    ham0 = MVP(5,5)
    ham0.calc_hmat(indmap)