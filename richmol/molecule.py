import autograd.numpy as np
from autograd import elementwise_grad, jacobian, grad
import functools
import sys
import constants


class Molecule():

    def __init__(self, *args, **kwargs):
        if 'masses' in kwargs:
            self.natoms = len(kwargs['masses'])
            self.masses = np.array(kwargs['masses'])
            self.totmass = np.sum(self.masses)
        if 'poten' in kwargs:
            self.fpot = kwargs['poten']


    def autograd(method):
        @functools.wraps(method)
        def wrapper_autograd(self, *args, **kwargs):
            xyz = method(self, *args, **kwargs)
            try:
                iatom = kwargs['atom']
            except KeyError:
                raise KeyError(f"atom=<atom number> argument is missing") from None
            try:
                ialpha = kwargs['alpha']
            except KeyError:
                raise KeyError(f"alpha=<Cartesian component> argument is missing") from None
            try:
                res = xyz[:,iatom,ialpha]
            except IndexError:
                raise IndexError(f"Cartesian component {ialpha} for atom number {iatom} is out of " \
                        +f"bounds for molecule type '{self.__class__.__name__}'") from None
            return res
        return wrapper_autograd


    def com(method):
        """ Shifts Cartesian coordinates to centre of mass """
        @functools.wraps(method)
        def wrapper_cm(self, *args, **kwargs):
            xyz = method(self, *args, **kwargs)
            # move to nuclear centre of mass
            com = np.dot(self.masses,xyz)/self.totmass
            return xyz - com[:,np.newaxis,:]
        return wrapper_cm


    def bisector(axes='zyx'):
        """For triatomic molecule rotates axes system to a bisector orthogonal frame

        Here, the origin is shifted to atom no. 0, and the mapping of xyz axes on a bisector frame
        is defined by 'axes' (str) argument.
        For example, axes='xyz' means that 'x' is a bisector, 'y' is along the axis perpendicular
        to a molecular frame, and 'z' lies in the molecular frame.
        """
        if "".join(sorted(axes.lower())) != "xyz":
            raise ValueError(f"wrong axes specification '{axes}' (must contain 'x', 'y', and 'z')")
        axes_ind = [("x", "y", "z").index(s) for s in list(axes.lower())]
        def inner_function(method):
            @functools.wraps(method)
            def wrapper_bisect(self, *args, **kwargs):
                xyz = method(self, *args, **kwargs)
                if xyz.shape[1] != 3:
                    raise RuntimeError(f"'bisector' decorator cannot be applied to " \
                            +f"'{self.__class__.__name__}.{method.__name__}'") from None
                r1 = np.linalg.norm(xyz[:,1,:] - xyz[:,0,:], axis=1)
                r2 = np.linalg.norm(xyz[:,2,:] - xyz[:,0,:], axis=1)
                e1 = (xyz[:,1,:] - xyz[:,0,:]) / r1[:,np.newaxis]
                e2 = (xyz[:,2,:] - xyz[:,0,:]) / r2[:,np.newaxis]
                n1 = (e1 + e2)
                l1 = np.linalg.norm(n1, axis=1)
                n1 = n1 / l1[:,np.newaxis]
                n2 = np.cross(e1, n1, axis=1)
                l2 = np.linalg.norm(n2, axis=1)
                n2 = n2 / l2[:,np.newaxis]
                n3 = np.cross(n1, n2, axis=1)
                l3 = np.linalg.norm(n3, axis=1)
                n3 = n3 / l3[:,np.newaxis]
                tmat = np.stack((n1,n2,n3), axis=1)
                return np.einsum('ijk,ilk->ijl', xyz, tmat[:,axes_ind,:])
            return wrapper_bisect
        return inner_function


    def G_invcm(method):
        """ Changes units of G-matrix to cm^{-1} """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            gmat = method(self, *args, **kwargs)
            to_invcm = constants.planck * constants.avogno * 1.0e+16 \
                     / (4.0 * np.pi * np.pi * constants.vellgt)

            return gmat * to_invcm
        return wrapper


    @G_invcm
    def G(self, coords):
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
        natoms =  self.natoms
        natoms3 = natoms * 3
        try:
            npoints = coords.shape[0]
            ncoords = coords.shape[1]
        except AttributeError:
            raise AttributeError(f"input array 'coords' is not a 'numpy.ndarray' type")
        except IndexError:
            raise IndexError(f"input array 'coords' has wrong number of dimensions = {coords.ndim} " \
                    +f"(expected 2)") from None

        # vibrational part

        xyz_grad = elementwise_grad(self.internal_to_cartesian)
        tvib = np.array([ xyz_grad(coords, atom = iatom, alpha = ialpha) \
                          for iatom in range(natoms) for ialpha in range(3) ]) # shape = (iatom*ialpha, ipoint, icoord)

        # rotational part

        eps = np.array([[[ int((ialpha - ibeta) * (ibeta - igamma) * (igamma - ialpha) / 2) \
                           for igamma in range(3) ] for ibeta in range(3) ] for ialpha in range(3) ])

        xyz = np.array([[ self.internal_to_cartesian(coords, atom = iatom, alpha = igamma) \
                          for igamma in range(3) ] for iatom in range(natoms) ]) # shape = (iatom, igamma, ipoint)

        trot = np.reshape(np.transpose(np.dot(eps, xyz), (0,2,3,1)), (natoms3, npoints, 3)) # shape = (iatom*ialpha, ipoint, icoord)

        # translational part

        ttra_ = np.reshape(np.array([np.eye(3, dtype=np.float64) for iatom in range(natoms)]), (natoms3, 3))
        ttra = np.zeros((npoints, natoms3, 3), dtype=np.float64)
        ttra[:,:,:] = ttra_[:,:]

        # join vibrational, rotational and translational parts

        tvec = np.concatenate((tvib, trot, ttra.transpose((1,0,2))), axis=2)

        # g-small matrix

        sqm = np.array([np.sqrt(self.masses[iatom]) for iatom in range(natoms) for ialpha in range(3)])
        tvec *= sqm[:, np.newaxis, np.newaxis]
        gsmall = np.einsum('ijk,ijl->jkl', tvec, tvec)

        # inverse of g-small = G-big

        gmat = np.linalg.inv(gsmall)
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
        to_invcm = constants.planck * constants.avogno * 1.0e+16 \
                 / (4.0 * np.pi * np.pi * constants.vellgt)

        def _determinant(coord):
            """
            Computes the determinant of the G matrix at a certain point
            """
            # consider only the rovibrational part of G
            G = self.G(coord)[:, 0:n_internal_coords+3, 0:n_internal_coords+3]
            #G /= to_invcm
            det = np.linalg.det(G)
            return det

        def _func(coordinate):
            """
            Computes: G/|G| . grad(|G|)
            """
            det = _determinant(coordinate)
            # compute gradient of the determinant
            grada_val = grada(coordinate).transpose()
            # vibrational part of G
            G = self.G(coordinate)[:, 0:n_internal_coords, 0:n_internal_coords]
            #G /= to_invcm
            m = G/det
            # method 1
            return np.dot(m[0,:,:],  grada_val)[:,0]
            #method 2
            #print(np.shape(np.multiply(m, grada_val)[0,:,:]))

            #return np.multiply(m, grada_val)[0,:,:]

        pseudo_poten = []
        n_internal_coords = np.shape(coords)[1]
        n_grid = np.shape(coords)[0]

        for icoord in range(n_grid):
            coordinate = coords[icoord, :].reshape(-1,1).transpose()

            #Compute first term of the PP
            # compute gradient of the determinant
            grada = grad(_determinant)
            grada_val = grada(coordinate).transpose()
            # vibrational G/ det^2
            G = self.G(coordinate)[:, 0:n_internal_coords, 0:n_internal_coords]
            #G /= to_invcm
            det = _determinant(coordinate)
            m = G/det**(2)
            #method 1
            part1 = np.dot(np.dot(m[0,:,:], grada_val).transpose(), grada_val)
            print(f"part1 using method 1 is: {part1}")
            #method 2:
            #part1 = np.dot(np.multiply(np.diagonal(m[0,:,:]), grada_val[:,0]),  grada_val)


        #    Compute second term of the PP

            grada_3 = jacobian(_func)
            grada_3_eval = grada_3(coordinate).reshape((n_internal_coords, n_internal_coords))
            part2 = 4*np.sum(np.diagonal(grada_3_eval))

            #method 2
            #grada_3_eval = grada_3(coordinate).reshape((n_internal_coords, n_internal_coords, n_internal_coords))
            #part2 = 4*np.sum(np.diagonal(np.diagonal(grada_3_eval)))

            U = (1/32)*(part1-part2)

            print(U)
            #to_invcm = constants.planck * constants.avogno * 1.0e+16 \
            #         / (4.0 * np.pi * np.pi * constants.vellgt)
            #pseudo_poten.append(U/(to_invcm))
            pseudo_poten.append(U)

            """
        for icoord in range(1):
            coordinate = coords[icoord, :].reshape(-1,1).transpose()
            det = _determinant(coordinate)
            G = self.G(coordinate)[:, 0:n_internal_coords, 0:n_internal_coords][0,:,:]
            grada = grad(_determinant)
            grada_val = grada(coordinate).transpose()
            print(np.shape(grada_val))
            part1 = 0.0
            for k in range(n_internal_coords):
                for l in range(n_internal_coords):
                    part1 += (G[k,l]/det**(2))*grada_val[k,0]*grada_val[l,0]
            print(f"part 1 using method 2 is: {part1}")
            sys.exit()
            """
        return np.array(pseudo_poten)
