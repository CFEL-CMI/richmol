import numpy as np
import Tasmanian
from numpy.polynomial.hermite import hermgauss


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

    G, _ = gmat(qref)

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


def gausshermite1d(lev, qind, qref, gmat, poten, fdn_step=0.001):

    x, w = hermgauss(lev)

    G, _ = gmat(qref)

    # 7-point finite-difference rule for 2-order derivative of potential
    h = fdn_step
    fdn_h = np.array([3*h, 2*h, h, 0.0, -h, -2*h, -3*h], dtype=np.float64)
    fdn_c = np.array([2.0, -27.0, 270.0, -490.0, 270.0, -27.0, 2.0], dtype=np.float64)
    fdn_d = 180.0

    # compute scaling
    q = np.array(np.broadcast_to(qref, (len(fdn_h), len(qref)))).T
    q[qind, :] = [qref[qind] + h for h in fdn_h]
    freq = np.dot(poten(*q), fdn_c) / (fdn_d * fdn_step**2)
    scaling = np.sqrt(np.sqrt( 2.0 * np.abs(freq) / np.abs(G[qind, qind]) ))

    points = x / scaling + qref[qind]
    points = np.array(np.broadcast_to(points, (1,len(points)))).T
    return points, w, scaling
