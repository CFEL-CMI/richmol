import expokit
import numpy as np


class ExpokitError(Exception):
    pass 

def check_exit_flag(iflag):
    if iflag < 0:
        raise ExpokitError("bad input arguments")
    elif iflag > 0:
        raise ExpokitError(
            { 1 : "maximum number of steps reached without convergence",
              2 : "requested tolerance was too high" }[iflag]
        )


def dsexpv(vec, norm, m, t, matvec, tol=0):
    """Computes exp(t*A)*v using Krylov, A symmetric

        Args:
            vec : numpy.ndarray
                Vector `v` to multiply matrix-exponential with
            norm : float
                Lower bound of the 1-norm of the sparse matrix `A`
            t : float
                Factor `t` in exponent
            matvec : lambda
                Matrix-vector product function. Takes an input vector of type
                'numpy.ndarray' as argument and returns the resulting vector
                of type 'numpy.ndarray'.

        Kwargs:
           tol : int, float
               TODO

        Returns:
            u : numpy.ndarray
                Resulting vector
    """

    n = vec.shape[0]
    v = vec.astype(np.float64, casting="safe").ravel()
    workspace = np.zeros(n*(m+2)+5*(m+2)*(m+2)+7, dtype=np.float64)
    iworkspace = np.zeros(m+2, dtype=np.int32)

    u, tol0, iflag = expokit.dsexpv(
        m, t, v, tol, norm, workspace, iworkspace, matvec, 0
    )

    check_exit_flag(iflag)
    return u


def zhexpv(vec, norm, m, t, matvec, tol=0):
    """Computes exp(t*A)*v using Krylov, A Hermitian

        Args:
            vec : numpy.ndarray
                Vector `v` to multiply matrix-exponential with
            norm : float
                Lower bound of the 1-norm of the sparse matrix `A`
            t : float
                Factor `t` in exponent
            matvec : lambda
                Matrix-vector product function. Takes an input vector of type
                'numpy.ndarray' as argument and returns the resulting vector
                of type 'numpy.ndarray'.

        Kwargs:
           tol : int, float
               TODO

        Returns:
            u : numpy.ndarray
                Resulting vector
    """

    n = vec.shape[0]
    v = vec.astype(np.complex128, casting="safe").ravel()
    workspace = np.zeros(n*(m+2)+5*(m+2)*(m+2)+7, dtype=np.complex128)
    iworkspace = np.zeros(m+2, dtype=np.int32)

    u, tol0, iflag = expokit.zhexpv(
        m, t, v, tol, norm, workspace, iworkspace, matvec, 0
    )

    check_exit_flag(iflag)
    return u
