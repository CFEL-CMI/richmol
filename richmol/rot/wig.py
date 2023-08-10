import numpy as np
from scipy.sparse import diags


def jy_eig(j: float, trid=True):
    assert (j>=0), f"j < 0: {j} < 0"
    x = lambda m: np.sqrt((j + m) * (j - m + 1))
    d = lambda m, n: 1 if m==n else 0

    if trid:
        # faster, using tridiagonal matrix
        ld = np.array([x(n) for n in np.linspace(-j, j, int(2*j+1))])
        ud = np.array([x(-n) for n in np.linspace(-j, j, int(2*j+1))])
        jmat = diags([ld[1:], np.zeros(2*j+1), ud[:-1]], [-1, 0, 1]).toarray() * 0.5j
    else:
        # slower, using dense matrix
        jmat = np.array([[0.5j * (x(-n) * d(m, n+1) - x(n) * d(m, n-1))
                            for n in np.linspace(-j, j, int(2*j+1))]
                            for m in np.linspace(-j, j, int(2*j+1))], dtype=np.complex128)
    e, v = np.linalg.eigh(jmat)
    return v


def wig_d(j, beta):
    v = jy_eig(j)
    e = np.exp(-1j * np.linspace(-j, j, int(2*j+1))[None, :] * beta[:, None])
    res = np.einsum('gi,mi,ki->mkg', e, np.conj(v), v, optimize='optimal')
    return res


def wig_D(j, alpha, beta, gamma):
    d = wig_d(j, beta)
    m = np.linspace(-j, j, int(2*j+1))
    k = np.linspace(-j, j, int(2*j+1))
    em = np.exp(1j * alpha[None, None, :] * m[:, None, None])
    ek = np.exp(1j * gamma[None, None, :] * k[None, :, None])
    return em * ek * d



if __name__ == "__main__":
    import spherical
    import quaternionic
    import itertools

    print("Test present approach for Wigner D-functions and compare with 'spherical'")

    alpha = np.linspace(0, 2*np.pi, 10)
    beta = np.linspace(0, np.pi, 10)
    gamma = np.linspace(0, 2*np.pi, 10)
    grid = np.array([elem for elem in itertools.product(alpha, beta, gamma)])
    alpha, beta, gamma = grid.T
    print("number of points:", len(grid))

    j = 100
    print("j =", j)

    print("wig_D ...")
    d = wig_D(j, alpha, beta, gamma)
    print("... done")

    print("spherical ...")
    wigner = spherical.Wigner(j)
    R = quaternionic.array.from_euler_angles(alpha, beta, gamma)
    wigD = wigner.D(R)
    d2 = np.array([[wigD[:, wigner.Dindex(int(j), int(m), int(k))]
                      for k in np.linspace(-j, j, int(2*j+1))]
                      for m in np.linspace(-j, j, int(2*j+1))])
    print("... done")

    print("max difference:", np.max(np.abs(d - d2)))
