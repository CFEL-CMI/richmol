import numpy as np
#import Tasmanian
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import sys

def gausshermite(lev, qind, qref, gmat, poten, sparse_type="qptotal", fdn_step=0.001):
    """Sparse grid using Gauss-Hermite rules, as implemented in Tasmanian

    Args:
        lev : int
            Level of grid, increasing level leads to growing number of points
        qind : list
            Indices of internal coordinates for grid
        qref : array
            Reference values of all internal coordinates
        gmat : function(*q)
            Kinetic energy matrix, function of all internal coordinates `q`
        poten : function(*q)
            Potential energy surface, function of all internal coordinates `q`

    Returns:
        points : array(no_points, len(qind))
            Grid points, `no_points` is the number of points
        weights : array(no_points)
            Quadrature weights
    """
    assert all(i >=0 and i <= len(qref) for i in qind), f"bogus coordinate indices in 'qind': {qind}"

    grid = Tasmanian.TasmanianSparseGrid()
    grid.makeGlobalGrid(len(qind), 0, lev, sparse_type, "gauss-hermite")

    # set coordinate domains
    # from Tasmanian manual: Sparse grids are build on canonical 1D domain [−1,1], with the exception
    #   of Gauss-Laguerre and Gauss-Hermite rules that use [0,∞) and (−∞,∞) respectively.
    #   Linear transformation can be applied to translate [−1,1] to an arbitrary interval [a,b],
    #   for unbounded domain we can apply shift a and scaling b

    # compute scaling parameter `scale` for each coordinate in `qind`

    G = gmat(qref)

    # 7-point finite-difference rule for 2-order derivative of potential
    h = fdn_step
    fdn_h = np.array([3*h, 2*h, h, 0.0, -h, -2*h, -3*h], dtype=np.float64)
    fdn_c = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdn_d = 180.0

    # compute scaling
    scaling = []
    for i in qind:
        q = np.array(np.broadcast_to(qref, (len(fdn_h), len(qref)))).T
        q[i, :] = [qref[i] + h for h in fdn_h]
        freq = np.dot(poten(*q), fdn_c) / (fdn_d * fdn_step**2)
        scaling.append( np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G[i, i]) )) )

    # apply shift and scaling to grid points

    # grid.setDomainTransform(np.array([(qref[i], 1) for i, b in zip(qind, scaling)]))

    points = grid.getPoints() / np.array(scaling) + np.array([qref[i] for i in qind])
    weights = grid.getQuadratureWeights()

    return points, weights, scaling


def herm1d(npt, ind, ref, gmat, poten, h=0.001):
    """One-dimensional Gauss-Hermite quadrature, shifted and scaled

    Args:
        npt : int
            Number of quadrature points
        ind : int
            Index of internal coordinate for quadrature
        ref : array (ncoords)
            Reference (equilibrium) values of all internal coordinates
        gmat : function(coords)
            Kinetic energy matrix, function of all internal coordinates `coords`
        poten : function(*coords)
            Potential energy, function of all internal coordinates `coords`

    Returns:
        points : array(npt)
            Quadrature points, points = x / scaling + ref[ind], where x are quadrature abscissas
        weights : array(npt)
            Quadrature weights
        scaling : float
            Quadrature scaling factor
    """
    x, weights = hermgauss(npt)

    G = gmat(ref)

    # 7-point finite-difference rule for 2-order derivative of potential
    fdn_step = np.array([3*h, 2*h, h, 0.0, -h, -2*h, -3*h], dtype=np.float64)
    fdn_coef = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdn_denom = 180.0

    # compute scaling
    coords = np.array(np.broadcast_to(ref, (len(fdn_step), len(ref)))).T
    coords[ind, :] = [ref[ind] + step for step in fdn_step]
    freq = np.dot(poten(*coords), fdn_coef) / (fdn_denom * h**2)
    scaling = np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G[ind, ind]) ))

    points = x / scaling + ref[ind]
    return points, weights, scaling

def legendre1d(npt, ind, a, b, ref):
    """One-dimensional Gauss-Hermite quadrature, shifted and scaled according to:
    r = 2/(b-a)*x + (b+a)/2

    Args:
        npt : int
            Number of quadrature points
        ind : int
            Index of internal coordinate for quadrature
        ref : array (ncoords)
            Reference (equilibrium) values of all internal coordinates
        gmat : function(coords)
            Kinetic energy matrix, function of all internal coordinates `coords`
        poten : function(*coords)
            Potential energy, function of all internal coordinates `coords`

    Returns:
        points : array(npt)
            Quadrature points, points = x / scaling + ref[ind], where x are quadrature abscissas
        weights : array(npt)
            Quadrature weights
        scaling : float
            Quadrature scaling factor
    """
    x, weights = leggauss(npt)
    scaling = (b-a)/2
    points = x / scaling + ref[ind]

    return points, weights, scaling

def prodgrid(quads, ind=None, ref=None, poten=None, pthr=None, wthr=None):
    """Direct product grid

    Args:
        quads : list of tuples (points, weights, scaling)
            List of one-dimensional quadratures, where each element of the list is a tuple
            containing quadrature points, weights, and scaling parameters
        ind : list
            Indices of internal coordinates corresponding to one-dimensional quadratures
        ref : array (ncoords)
            Reference (equilibrium) values of all internal coordinates
        poten : function(*coords)
            Potential energy, function of all internal coordinates `coords`
        pthr : float
            Threshold for neglecting points with potential larger than `thresh`
        wthr : float
            Threshold for neglecting points with total quadrature weight smaller than `wthr`
    """
    if ind is not None:
        assert (len(quads) == len(ind)), f"size of `ind` = {len(ind)} is not equal to size of `quad` = {len(quad)}"
    else:
        ind = [i for i in range(len(quads))]
    if ref is not None:
        assert (all(i <= len(ref)-1 for i in ind)), f"some of coordinate indices in `ind` = {ind} exceed size of `ref` = {len(ref)}"
        ncoords = len(ref)
    else:
        ncoords = len(ind)

    points = (quads[i][0] if i in ind else [ref[i]] for i in range(ncoords))
    weights = (quads[i][1] for i in range(len(quads)))
    points = np.array(np.meshgrid(*points)).T.reshape(-1, ncoords)
    weights = np.array(np.meshgrid(*weights)).T.reshape(-1, len(quads))

    # remove points with large potential
    if pthr is not None:
        pot = poten(*points.T)
        pmin = np.min(pot)
        ind = np.where(pot - pmin < pthr)
        points = points[ind]
        weights = weights[ind]

    # remove points with small weight
    if wthr is not None:
        ind = np.where(np.prod(weights, axis=1) > wthr)
        points = points[ind]
        weights = weights[ind]

    return points, weights, [quad[2] for quad in quads]
