#import autograd.numpy as np
import jax.numpy as np
#from autograd import elementwise_grad, jacobian, grad
from jax import grad,  jacfwd, jacrev, jit
import functools
import sys
import constants
from jax.config import config
config.update("jax_enable_x64", True)

levi_chivita = np.array([[[ int((i - j) * (j - k) * (k - i) * 0.5) \
                            for k in range(3) ] for j in range(3) ] for i in range(3) ], \
                            dtype=np.float64)


class Molecule():

    def __init__(self, *args, **kwargs):
        if 'masses' in kwargs:
            self.natoms = len(kwargs['masses'])
            self.masses = np.array(kwargs['masses'])
            self.totmass = np.sum(self.masses)
        if 'poten' in kwargs:
            self.fpot = kwargs['poten']


    def elementwise(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            mat = method(self, *args, **kwargs)
            ndim = mat.ndim
            try:
                element_list = kwargs['elem']
                res = mat[np.ix_(*elem)]
            except KeyError:
                res = mat
            return res
        return wrapper


    def com(method):
        """ Shifts frame origin to the centre of mass """
        @functools.wraps(method)
        def wrapper_cm(self, *args, **kwargs):
            xyz = method(self, *args, **kwargs)
            com = np.dot(self.masses, xyz) / self.totmass
            return xyz - com[np.newaxis,:]
        return wrapper_cm


    def bisector(axes='zyx'):
        """For triatomic molecule rotates axes system to a bisector orthogonal frame

        Here, the origin is shifted to atom no. 0, and the mapping of xyz axes
        on a bisector frame is defined by 'axes' (str) argument.
        For example, axes='xyz' means that 'x' is a bisector, 'y' is along the axis perpendicular
        to a molecular frame, and 'z' lies in the molecular frame.
        """
        if "".join(sorted(axes.lower())) != "xyz":
            raise ValueError(f"wrong axes specification '{axes}' " \
                            +f"(must contain 'x', 'y', and 'z')") from None
        axes_ind = [("x", "y", "z").index(s) for s in list(axes.lower())]
        def inner_function(method):
            @functools.wraps(method)
            def wrapper_bisect(self, *args, **kwargs):
                xyz = method(self, *args, **kwargs)
                if xyz.shape[1] != 3:
                    raise RuntimeError(f"'bisector' decorator cannot be applied to " \
                            +f"'{self.__class__.__name__}.{method.__name__}'") from None
                r1 = np.linalg.norm(xyz[1,:] - xyz[0,:])
                r2 = np.linalg.norm(xyz[2,:] - xyz[0,:])
                e1 = (xyz[1,:] - xyz[0,:]) / r1
                e2 = (xyz[2,:] - xyz[0,:]) / r2
                n1 = (e1 + e2)
                l1 = np.linalg.norm(n1)
                n1 = n1 / l1
                n2 = np.cross(e1, n1)
                l2 = np.linalg.norm(n2)
                n2 = n2 / l2
                n3 = np.cross(n1, n2)
                l3 = np.linalg.norm(n3)
                n3 = n3 / l3
                tmat = np.stack((n1,n2,n3))
                return np.einsum('jk,lk->jl', xyz, tmat[axes_ind,:])
            return wrapper_bisect
        return inner_function


    def G_invcm(method):
        """ Changes units of kinetic energy operators to cm^{-1} """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            res = method(self, *args, **kwargs)
            to_invcm = constants.planck * constants.avogno * 1.0e+16 \
                     / (4.0 * np.pi * np.pi * constants.vellgt)

            return res * to_invcm
        return wrapper

    @com
    @bisector('zyx')
    def internal_to_cartesian(self, coords):
        r1, r2, alpha = coords
        xyz = np.array([[0.0, 0.0, 0.0], \
                        [ r1 * np.sin(alpha/2), 0.0, r1 * np.cos(alpha/2)], \
                        [-r2 * np.sin(alpha/2), 0.0, r2 * np.cos(alpha/2)]], \
                        dtype=np.float64)
        return xyz



    @jit
    def g(self, coords, **kwargs):
        natoms =  self.natoms
        natoms3 = natoms * 3

        tvib = np.array(jacfwd(self.internal_to_cartesian)(coords)) # shape = (iatom, ialpha, icoord_vib)
        xyz = self.internal_to_cartesian(coords)
        trot = np.transpose(np.dot(levi_chivita, xyz.T), (2,0,1)) # shape = (iatom, ialpha, icoord_rot)
        ttra = np.array([np.eye(3, dtype=np.float64) for iatom in range(natoms)])

        tvec = np.concatenate((tvib, trot, ttra), axis=2)
        sqm = np.array([np.sqrt(self.masses[iatom]) for iatom in range(natoms)])
        tvec = tvec * sqm[:, np.newaxis, np.newaxis]

        gsmall = np.einsum('ijk,ijl->kl', tvec, tvec)
        return gsmall


    @G_invcm
    def G(self, coords, **kwargs):
        """G-matrix using Autograd derivatives

        Args:
            coords (array (no_points, no_coords)): Values of internal coordinates on grid,
                no_coords is the number of internal coordinates,
                no_points is the number of grid points.

        Returns:
            gmat (array (no_points, no_coords+6, no_coords+6)): Elements of G-matrix on grid,
                coordinate index in range(no_coords), range(no_coords,no_coords+3), and
                range(no_coords+3,no_coords+6) corresponds to internal vibrational, three rotational,
                and three translational coordinates, respectively.
        """

        # inverse of g-small = G-big
        gmat = np.linalg.inv(self.g(coords))
        return gmat


    def V(self, coords):
        """Potential energy surface

        Args:
            coords (array (no_points, no_coords)): Values of internal coordinates on grid,
                no_coords is the number of internal coordinates,
                no_points is the number of grid points.

        Returns:
            poten (array (no_points)): Potential energy on grid.
        """
        return self.fpot(coords)

    def PP(self, coords):

        n_grid_points = np.shape(coords)[0]
        n_coords = np.shape(coords)[1]

        """Pseudo potential using Autograd derivatives

        Args:
            coords (array (no_points, no_coords)): Values of internal coordinates on grid,
                no_coords is the number of internal coordinates,
                no_points is the number of grid points.

        Returns:
            pseudo_poten (array (no_points,)): the pseudo potential on each point of the grid

        Side notes:
        This should still be vectorized for the coordinate choice
        multiply by hbar?
    """
        def _determinant(coordinate):
            G  = self.G(coordinate)[0,0:n_coords+6,0:n_coords+6]
            # There is a big difference in including translations or not
            det = np.linalg.det(G)
            return det

        def _func(coordinate):
            G  = self.G(coordinate)[0,0:n_coords,0:n_coords]
            Det = _determinant(coordinate)
            dDet = grad(_determinant)
            dDet_val = dDet(coordinate)
            f = np.dot(G, dDet_val.transpose())/Det
            return f
        n_grid_points = np.shape(coords)[0]
        n_coords = np.shape(coords)[1]
        pseudos = []

        for icoord in range(n_grid_points):
            coordinate = coords[icoord,:].reshape(1,-1)
            G = self.G(coordinate)[0,0:n_coords,0:n_coords]
            Det = _determinant(coordinate)
            dDet = grad(_determinant)
            dDet_val = dDet(coordinate)
            pseudo1 = np.dot(np.dot(dDet_val, G), dDet_val.transpose())/Det**2

            pseudo2 = 0.0
            f = _func(coordinate)
            grad_2 = jacobian(_func)

            grad_2_val = grad_2(coordinate)
            pseudo_2 = 4*np.sum(np.diagonal(grad_2_val))

            pseudo = (pseudo1 - pseudo_2)/32
            print(pseudo)
            pseudos.append(pseudo)
        #return np.array(pseudo_poten)

        return np.array(pseudos)[:,0,0]

    def dG(self, coords):
        ncoords = coords.shape[1]
        g_grad = elementwise_grad(self.g)
        grad = np.array([[ g_grad(coords, i = i, j = j) for i in range(9)] \
                          for j in range(9) ]) # shape = (i, j, ipoint, icoord)
        Gmat = self.G(coords)
        Gmat = np.einsum('ijk,ikl->ijl', Gmat, Gmat)
        print(grad.shape, Gmat.shape)
        return -np.einsum('ijk,klin->ijln', Gmat, grad)


    def G_d(self, coords):

        n_grid_points = np.shape(coords)[0]
        n_coords = np.shape(coords)[1]

        def _G(coordinate):
            G = self.G(coordinate)
            return G

        # try vectorizing over the grid points
        """
        dG = jacobian(_G)
        dG_val = dG(coords)
        print(np.shape(dG_val))
        dG_val.reshape((n_grid_points, n_coords+6,n_coords+6, 3))
        """
        dG_vals = []

        for ipoint in range(n_grid_points):
            coordinate = coords[ipoint,:].reshape(1,-1)
            dG = jacobian(_G)
            dG_val = dG(coordinate).reshape((n_coords+6,n_coords+6, 3))
            dG_vals.append(dG_val)
        print(np.shape(np.array(dG_vals)))

        return dG_vals
