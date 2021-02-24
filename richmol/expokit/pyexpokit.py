import expokit
import numpy as np

class ExpokitError(Exception):
    pass 

def check_exit_flag(iflag):
    if iflag<0:
        raise ExpokitError("bad input arguments")
    elif iflag>0:
        raise ExpokitError({1:"maximum number of steps reached without convergence", \
            2:"requested tolerance was too high"}[iflag])


def dsexpv(vec, norm, m, t, matvec, tol=0):
    """ Computes exp(t*A)*v using Krylov, A symmetric """

    n = vec.shape[0]
    v = vec.astype(np.float64, casting="safe").ravel()
    workspace = np.zeros(n*(m+2)+5*(m+2)*(m+2)+7, dtype=np.float64)
    iworkspace = np.zeros(m+2, dtype=np.int32)

    u, tol0, iflag = expokit.dsexpv(m, t, v, tol, norm, workspace, iworkspace, matvec, 0)

    check_exit_flag(iflag)
    return u


def zhexpv(vec, norm, m, t, matvec, tol=0):
    """ Computes exp(t*A)*v using Krylov, A Hermitian """

    n = vec.shape[0]
    v = vec.astype(np.complex128, casting="safe").ravel()
    workspace = np.zeros(n*(m+2)+5*(m+2)*(m+2)+7, dtype=np.complex128)
    iworkspace = np.zeros(m+2, dtype=np.int32)

    u, tol0, iflag = expokit.zhexpv(m, t, v, tol, norm, workspace, iworkspace, matvec, 0)

    check_exit_flag(iflag)
    return u


if __name__ == "__main__":

    from scipy.sparse import random
    from scipy.sparse.linalg import onenormest, expm

    # generate random real symmetric matrix and vector
    n = 1000
    m = 12
    vec = np.random.ranf(n)
    vec = vec / np.linalg.norm(vec)
    mat = random(n, n, format="csr")
    mat = (mat.T + mat) / 2

    # use expokit
    norm = onenormest(mat)
    matvec = lambda v: mat.dot(v)
    time = 1.0
    res = dsexpv(vec, norm, m, time, matvec)

    # use expm method (Pade approximation)
    matexp = expm(time*mat)
    res2 = matexp.dot(vec)

    print("maximal difference 'dsexpv' - 'expm':", np.max(np.abs(res2-res)))

    # same check for complex matrix

    n = 1000
    m = 12
    vec = np.random.ranf(n)
    vec = vec / np.linalg.norm(vec)
    mat = random(n, n, format="csr") + random(n, n, format="csr")*1j
    mat = (np.conjugate(mat.T) + mat) / 2

    # use expokit
    norm = onenormest(mat)
    matvec = lambda v: mat.dot(v)
    time = 1.0
    res = zhexpv(vec, norm, m, time, matvec)

    # use expm method (Pade approximation)
    matexp = expm(time*mat)
    res2 = matexp.dot(vec)

    print("maximal difference 'zhexpv' - 'expm':", np.max(np.abs(res2-res)))
