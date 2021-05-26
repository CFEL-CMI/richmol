import expokit
import numpy as np
import copy
from scipy.sparse.linalg import expm
from numba import njit, prange, cuda, int32, float64, complex128
from numba import cuda, complex128

import torch # temporary fix
if torch.cuda.is_available():
    import cupy as cp
    import cupyx


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


def expv_taylor(vec, m, t, matvec, maxorder=1000, tol=0):
    """ Computes epx(t*a)*v using Taylor with cupy mat-vec product """

    V = []

    # zeroth order
    V.append(vec)

    # higher orders
    conv_k, k = 1, 0
    while k < maxorder and conv_k > tol:
        k += 1
        v = matvec(V[k - 1]) * t / k / 2**k
        conv_k = np.sum(np.abs(v)**2)
        V.append(v)

    if k == maxorder:
        print("taylor reached maximum order of {}".format(maxorder))

    u = sum(V)

    return u


def expv_arnoldi(vec, m, t, matvec, maxorder=1000, tol=0):
    """ Computes exp(t*a)*v using Arnodli with cupy mat-vec product """

    V = []
    H = np.zeros((maxorder, maxorder), dtype=vec.dtype)

    # first Krylov basis vector
    V.append(vec)

    # higher orders
    u_kminus1, u_k, conv_k, k = {}, V[0], 1, 1
    while k < maxorder and conv_k > tol:

        # extend ONB of Krylov subspace by another vector
        v = matvec(V[k - 1])
        for j in range(k):
            H[j, k - 1] = np.vdot(V[j], v)
            v = v - H[j, k - 1] * V[j]

        # calculate current approximation and convergence
        u_kminus1 = u_k
        expH_k = expm(t * H[: k, : k])
        u_k = sum([expH_k[i, 0] * v_i for i,v_i in enumerate(V)])
        conv_k = np.sum(np.abs(u_k - u_kminus1)**2)

        # stop if new vector vanishes, else normalize
        H[k, k - 1] = np.sqrt(np.sum(np.abs(v)**2))
        if H[k, k - 1] < tol:
            break

        V.append(v / H[k, k -1])
        k += 1

    if k == maxorder:
        print("arnoldi reached maximum order of {}".format(maxorder))

    return u_k


def expv_lanczos(vec, m, t, matvec, maxorder=1000, tol=0):
    """ Computes epx(t*a)*v using Lanczos with cupy mat-vec product """

    V, W = [], []
    T = np.zeros((maxorder, maxorder), dtype=vec.dtype)

    # first Krylov basis vector
    V.append(vec)
    w = matvec(V[0])
    T[0, 0] = np.vdot(w, V[0])
    W.append(w - T[0, 0] * V[0])

    # higher orders
    u_kminus1, u_k, conv_k, k = {}, V[0], 1, 1
    while k < maxorder and conv_k > tol:

        # extend ONB of Krylov subspace by another vector
        T[k - 1, k] = np.sqrt(np.sum(np.abs(W[k - 1])**2))
        T[k, k - 1] = T[k - 1, k]
        if not T[k - 1, k] == 0:
            V.append(W[k - 1] / T[k - 1, k])

        # reorthonormalize ONB of Krylov subspace, if neccesary
        else:
            v = np.ones(V[k - 1].shape, dtype=np.complex128)
            for j in range(k):
                proj_j = np.vdot(V[j], v)
                v = v - proj_j * V[j]
            norm_v = np.sqrt(np.sum(np.abs(v)**2))
            V.append(v / norm_v)

        w = matvec(V[k])
        T[k, k] = np.vdot(w, V[k])
        w = w - T[k, k] * V[k] - T[k - 1, k] * V[k - 1]
        W.append(w)

        # calculate current approximation and convergence
        u_kminus1 = u_k
        expT_k = expm(t * T[: k + 1, : k + 1])
        u_k = sum([expT_k[i, 0] * v_i for i,v_i in enumerate(V)])
        conv_k = np.sum(np.abs(u_k - u_kminus1)**2)

        k += 1

    if k == maxorder:
        print("lanczos reached maximum order of {}".format(maxorder))

    return u_k


def matvec_numpy(matrix, counter):
    """ Computes lambda function for mat-vec prodcut with numba """
    # define mat-vec product with numpy
    def matvec(v, counter_):
        u = matrix.dot(v)
        counter_ += 1
        return u

    # create lambda function of above mat-vec product
    matvec_ = lambda v : matvec(v, counter)

    return matvec_


def matvec_numba(matrix):
    """ Computes lambda function for mat-vec prodcut with numba """
    # get components of sparse matrix
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr

    # define mat-vec product with numba
    @njit(parallel=True)
    def matvec(v):
        u = np.zeros(v.size, dtype=v.dtype)
        for i in prange(v.size):
            for j in prange(indptr[i], indptr[i + 1]):
                u[i] += data[j] * v[indices[j]]
        return u

    # create lambda function of above mat-vec product
    matvec_ = lambda v : matvec(v)

    return matvec_


def matvec_cupy(matrix):
    """ Computes lambda function for mat-vec product with cupy """
    # load matrix onto GPU
    matrix_gpu = cupyx.scipy.sparse.csr_matrix(matrix)

    # define mat-vec product with cupy
    def matvec(v):
        u = matrix_gpu.dot(cp.array(v))
        return cp.asnumpy(u)

    # create lambda function of above mat-vec product
    matvec_ = lambda v : matvec(v)

    return matvec_


def matvec_cupy2(matrix):
    """ Computes lambda function for mat-vec product with cupy """
    # load matrix onto GPUs
    ind = int(matrix.shape[0] / 2)
    with cp.cuda.Device(0):
        matrix_gpu0 = cupyx.scipy.sparse.csr_matrix(matrix[: ind])
    with cp.cuda.Device(1):
        matrix_gpu1 = cupyx.scipy.sparse.csr_matrix(matrix[ind :])

    # define mat-vec product with cupy
    def matvec(v):
        with cp.cuda.Device(0):
            u0 = cp.asnumpy(matrix_gpu0.dot(cp.array(v)))
        with cp.cuda.Device(1):
            u1 = cp.asnumpy(matrix_gpu1.dot(cp.array(v)))
        return np.concatenate((u0, u1))

    # create lambda function of above mat-vec product
    matvec_ = lambda v : matvec(v)

    return matvec_


def matvec_cuda(matrix):
    """ Computes lambda function for mat-vec product with cuda """
    # get components of sparse matrix
    data_ = matrix.data
    indices_ = matrix.indices
    indptr_ = matrix.indptr

    def matvec_pre(v):
        u_ = np.zeros(v.size, dtype=v.dtype)
        # = cuda.to_device(u_)
        matvec[v.size, 256](data_, indices_, indptr_, u_, v)
        #return u.copy_to_host()
        return u_


    @cuda.jit
    def matvec(data, indices, indptr, u, v):

        s_solution = cuda.shared.array(shape=(256,), dtype=float64)
        s_indptr = cuda.shared.array(shape=(2,), dtype=int32)

        # x position of this thread/block in block/grid
        xBlock = cuda.threadIdx.x
        xGrid = cuda.blockIdx.x
        blockDim = cuda.blockDim.x
        # load index pointers into shared memory
        if xBlock <= 1:
            s_indptr[xBlock] = indptr[xGrid + xBlock]

        cuda.syncthreads()

        # compute partial mat-vec product without further use of shared memory
        i = s_indptr[0] + xBlock
        temp = 0
        while i < s_indptr[1]:
            temp += data[i] * v[indices[i]]
            i += blockDim
        s_solution[xBlock] = temp

        cuda.syncthreads()

        # sum up partial mat-vec products
        if xBlock == 0:
            temp = 0
            for j in range(blockDim):
                temp += s_solution[j]
            u[xGrid] = temp

    #create lambda function of above mat-vec product
    matvec_ = lambda v : matvec_pre(v)

    return matvec_


def matvec_cuda2(matrix):
    """ Computes lambda function for mat-vec product with cuda """
    # get components of sparse matrix
    data_ = matrix.data
    indices_ = matrix.indices
    indptr_ = matrix.indptr

    def matvec_pre(v):
        u_ = np.zeros(v.size, dtype=v.dtype)
        u = cuda.to_device(u_)
        matvec[v.size, 256](data_, indices_, indptr_, u, v)
        return u.copy_to_host()


    @cuda.jit
    def matvec(data, indices, indptr, u, v):

        s_indptr = cuda.shared.array(shape=(2,), dtype=int32)
        s_data = cuda.shared.array(shape=(256, 2), dtype=float64)
        s_hit = cuda.shared.array(shape=(256,), dtype=int32)

        # x position of this thread/block in block/grid
        xBlock = cuda.threadIdx.x
        xGrid = cuda.blockIdx.x
        blockDim = cuda.blockDim.x

       # load index pointers into shared memory
        if xBlock <= 1:
            s_indptr[xBlock] = indptr[xGrid + xBlock]
            s_hit[blockDim - 1 - xBlock] = 1

        cuda.syncthreads()

        # compute partial mat-vec product without further use of shared memory
        i = s_indptr[0] + xBlock
        temp = 0
        while s_hit[blockDim - 1] == 1:
            if i < s_indptr[1]:
                s_data[xBlock, 0] = data[i]
                s_data[xBlock, 1] = v[indices[i]]
                s_hit[xBlock] = 1
            else:
                s_hit[xBlock] = 0
            cuda.syncthreads()
            if xBlock == 0:
                for j in range(blockDim):
                    if s_hit[j] == 1:
                        temp += s_data[j, 0] * s_data[j, 1]
                    else:
                        break
            cuda.syncthreads()
            i += blockDim

        if xBlock == 0:
            u[xGrid] = temp

    #create lambda function of above mat-vec product
    matvec_ = lambda v : matvec_pre(v)

    return matvec_


def main_01():
    """ Test with random sparse matrices generated with scipy """

    # testing parameters
    iterations_per_matrix = 10
    tolerance = 1e-15


    # start testing
    algorithm_runtimes = {
        'real symmetric matrix' : {
            'expokit' : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'taylor'  : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'arnoldi' : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'lanczos' : {'numpy' : [], 'numba' : [], 'cupy' : []}
        },
        'conjugate symmetric matrix' : {
            'expokit' : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'taylor'  : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'arnoldi' : {'numpy' : [], 'numba' : [], 'cupy' : []},
            'lanczos' : {'numpy' : [], 'numba' : [], 'cupy' : []}
        }
    }
    package_runtimes = {
        'real symmetric matrix' : {
            'scipy' : [],
            'numpy' : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []},
            'numba' : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []},
            'cupy'  : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []}
        },
        'conjugate symmetric matrix' : {
            'scipy' : [],
            'numpy' : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []},
            'numba' : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []},
            'cupy'  : {'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []}
        }
    }
    counters = {
        'real symmetric matrix' : {
            'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []
        },
        'conjugate symmetric matrix' : {
            'expokit' : [], 'taylor' : [], 'arnoldi' : [], 'lanczos' : []
        }
    }
    #matrix_sizes = np.linspace(100, 1000, num=11, endpoint=True)
    matrix_sizes = [10000]
    #print("\n")
    for matrix_size in matrix_sizes:


        # check real symmetric matrix
        print("  REAL SYMMETRIC MATRIX", "\n")

        
        # generate random real symmetric matrix and vector
        vector = np.random.ranf(int(matrix_size))
        vector = vector / np.linalg.norm(vector)
        matrix = random(int(matrix_size), int(matrix_size), format="csr")
        matrix = (matrix.T + matrix) / 2
        t = -1e-0
        counter = np.zeros(1)

        # use scipy (Pade approximation)
        #result_scipy = expm(t * matrix).dot(vector)
        result_scipy = 0
        function = lambda : expm(t * matrix).dot(vector)
        #runtime = timeit.timeit(function, number=1)
        runtime = 0
        package_runtimes['real symmetric matrix']['scipy'].append(runtime)
        print("SCIPY                               : {} seconds".format(runtime))

        # use expokit with numpy mat-vec product
        norm = onenormest(matrix)
        matvec_np = matvec_numpy(matrix, counter)
        m = 9
        result_scipy = dsexpv(vector, norm, m, t, matvec_np)
        counters['real symmetric matrix']['expokit'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result_scipy)**2))
        function = lambda : dsexpv(vector, norm, m, t, matvec_np)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['expokit']['numpy'].append(runtime)
        package_runtimes['real symmetric matrix']['numpy']['expokit'].append(runtime)
        print("EXPOKIT (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use expokit with numba mat-vec product
        matvec_nu = matvec_numba(matrix)
        result = dsexpv(vector, norm, m, t, matvec_nu)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : dsexpv(vector, norm, m, t, matvec_nu)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['expokit']['numba'].append(runtime)
        package_runtimes['real symmetric matrix']['numba']['expokit'].append(runtime)
        print("EXPOKIT (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use expokit with cupy mat-vec product
        matvec_cp = matvec_cupy(matrix)
        result = dsexpv(vector, norm, m, t, matvec_cp)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : dsexpv(vector, norm, m, t, matvec_cp)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['expokit']['cupy'].append(runtime)
        package_runtimes['real symmetric matrix']['cupy']['expokit'].append(runtime)
        print("EXPOKIT (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use taylor with numpy mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_np, tol=tolerance)
        counters['real symmetric matrix']['taylor'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['taylor']['numpy'].append(runtime)
        package_runtimes['real symmetric matrix']['numpy']['taylor'].append(runtime)
        print("TAYLOR  (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use taylor with numba mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['taylor']['numba'].append(runtime)
        package_runtimes['real symmetric matrix']['numba']['taylor'].append(runtime)
        print("TAYLOR  (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use taylor with cupy mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['taylor']['cupy'].append(runtime)
        package_runtimes['real symmetric matrix']['cupy']['taylor'].append(runtime)
        print("TAYLOR  (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use arnoldi with numpy mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
        counters['real symmetric matrix']['arnoldi'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['arnoldi']['numpy'].append(runtime)
        package_runtimes['real symmetric matrix']['numpy']['arnoldi'].append(runtime)
        print("ARNOLDI (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use arnoldi with numba mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['arnoldi']['numba'].append(runtime)
        package_runtimes['real symmetric matrix']['numba']['arnoldi'].append(runtime)
        print("ARNOLDI (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use arnoldi with cupy mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['arnoldi']['cupy'].append(runtime)
        package_runtimes['real symmetric matrix']['cupy']['arnoldi'].append(runtime)
        print("ARNOLDI (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use lanczos with numpy mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
        counters['real symmetric matrix']['lanczos'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['lanczos']['numpy'].append(runtime)
        package_runtimes['real symmetric matrix']['numpy']['lanczos'].append(runtime)
        print("LANCZOS (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use lanczos with numba mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['lanczos']['numba'].append(runtime)
        package_runtimes['real symmetric matrix']['numba']['lanczos'].append(runtime)
        print("LANCZOS (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use lanczos with cupy mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['real symmetric matrix']['lanczos']['cupy'].append(runtime)
        package_runtimes['real symmetric matrix']['cupy']['lanczos'].append(runtime)
        print("LANCZOS (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))
        print("\n")


        sys.exit()
        # check conjugate symmetric matrix
        print("  COMPLEX HERMITIAN MATRIX", "\n")


        # generate random conjugate symmetric matrix and vector
        vector = np.random.ranf(int(matrix_size)) + np.random.ranf(int(matrix_size))*1j
        vector = vector / np.linalg.norm(vector)
        matrix = random(int(matrix_size), int(matrix_size), format="csr") + random(int(matrix_size), int(matrix_size), format="csr")*1j
        matrix = (np.conjugate(matrix.T) + matrix) / 2
        t = -1e-6


        # use scipy (Pade approximation)
        #result_scipy = expm(t * matrix).dot(vector)
        result_scipy = 0
        function = lambda : expm(t * matrix).dot(vector)
        #runtime = timeit.timeit(function, number=1)
        runtime = 0
        package_runtimes['conjugate symmetric matrix']['scipy'].append(runtime)
        print("SCIPY                               : {} seconds".format(runtime))


        # use expokit with numpy mat-vec product
        norm = onenormest(matrix)
        matvec_np = matvec_numpy(matrix, counter)
        m = 9
        result_scipy = zhexpv(vector, norm, m, t, matvec_np)
        counters['conjugate symmetric matrix']['expokit'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result_scipy)**2))
        function = lambda : zhexpv(vector, norm, m, t, matvec_np)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['expokit']['numpy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numpy']['expokit'].append(runtime)
        print("EXPOKIT (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use expokit with numba mat-vec product
        matvec_nu = matvec_numba(matrix)
        result = zhexpv(vector, norm, m, t, matvec_nu)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : zhexpv(vector, norm, m, t, matvec_nu)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['expokit']['numba'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numba']['expokit'].append(runtime)
        print("EXPOKIT (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use expokit with cupy mat-vec product
        matvec_cp = matvec_cupy(matrix)
        result = zhexpv(vector, norm, m, t, matvec_cp)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : zhexpv(vector, norm, m, t, matvec_cp)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['expokit']['cupy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['cupy']['expokit'].append(runtime)
        print("EXPOKIT (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use taylor with numpy mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_np, tol=tolerance)
        counters['conjugate symmetric matrix']['taylor'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['taylor']['numpy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numpy']['taylor'].append(runtime)
        print("TAYLOR  (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use taylor with numba mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['taylor']['numba'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numba']['taylor'].append(runtime)
        print("TAYLOR  (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use taylor with cupy mat-vec product
        result = expv_taylor(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_taylor(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['taylor']['cupy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['cupy']['taylor'].append(runtime)
        print("TAYLOR  (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use arnoldi with numpy mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
        counters['conjugate symmetric matrix']['arnoldi'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['arnoldi']['numpy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numpy']['arnoldi'].append(runtime)
        print("ARNOLDI (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use arnoldi with numba mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['arnoldi']['numba'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numba']['arnoldi'].append(runtime)
        print("ARNOLDI (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use arnoldi with cupy mat-vec product
        result = expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['arnoldi']['cupy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['cupy']['arnoldi'].append(runtime)
        print("ARNOLDI (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


        # use lanczos with numpy mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
        counters['conjugate symmetric matrix']['lanczos'].append(counter[0])
        counter[0] = 0
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['lanczos']['numpy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numpy']['lanczos'].append(runtime)
        print("LANCZOS (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use lanczos with numba mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['lanczos']['numba'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['numba']['lanczos'].append(runtime)
        print("LANCZOS (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

        # use lanczos with cupy mat-vec product
        result = expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
        deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
        function = lambda : expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        algorithm_runtimes['conjugate symmetric matrix']['lanczos']['cupy'].append(runtime)
        package_runtimes['conjugate symmetric matrix']['cupy']['lanczos'].append(runtime)
        print("lanczos (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))
        print("\n")
    sys.exit()

    # plot the results
    plot_dir = "t_{}_mat_size_{}_{}".format(t, min(matrix_sizes), max(matrix_sizes))
    os.mkdir(plot_dir)
    colors, linestyles = ["k", "b", "g", "r"], ["-", "--", ":"]
    """
    # one plot for each algorithm
    for mat_type in algorithm_runtimes.keys():
        for i, items in enumerate(algorithm_runtimes[mat_type].items()):
            alg_type, runtimes = items
            plt.title("{} & {} algorithm".format(mat_type, alg_type))
            for j, items_ in enumerate(runtimes.items()):
                pack_type, runtimes_ = items_
                plt.plot(matrix_sizes, np.array(runtimes_), label=pack_type, color=colors[i], linestyle=linestyles[j])
            plt.xlabel("matrix size (along one dimension)")
            plt.xlim([np.amin(matrix_sizes), np.amax(matrix_sizes)])
            plt.ylabel("runtime (sec)")
            plt.yscale("log")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}_{}.pdf".format(mat_type, alg_type)), format="pdf")
            plt.close()
    """
    # one plot for each python package
    for mat_type in package_runtimes.keys():
        for i, items in enumerate(package_runtimes[mat_type].items()):
            pack_type, runtimes = items
            plt.title("{} & {} mat-vec product".format(mat_type, pack_type))
            if pack_type == "scipy":
                #plt.plot(matrix_sizes, np.array(runtimes), label=pack_type, color='k')
                continue
            else:
                for j, items_ in enumerate(runtimes.items()):
                    alg_type, runtimes_ = items_
                    plt.plot(matrix_sizes, np.array(runtimes_), label=alg_type, color=colors[j], linestyle=linestyles[i - 1])
            plt.xlabel("matrix size (along one dimension)")
            plt.xlim([np.amin(matrix_sizes), np.amax(matrix_sizes)])
            plt.ylabel("runtime (sec)")
            plt.yscale("log")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, "{}_{}_t{}.pdf".format(mat_type, pack_type, t)), format="pdf")
            plt.close()

    # plot matvec function call counts
    for mat_type in counters.keys():
        plt.title("matvec call counts for {}".format(mat_type))
        for i, items in enumerate(counters[mat_type].items()):
            alg_type, call_counts = items
            plt.plot(matrix_sizes, np.array(counters[mat_type][alg_type]), label=alg_type, color=colors[i])
        plt.xlabel("matrix size (along one dimension)")
        plt.xlim([np.amin(matrix_sizes), np.amax(matrix_sizes)])
        plt.ylabel("matvec call count")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "matvec_call_count_{}_{}_{}.pdf".format(mat_type, pack_type, t)), format="pdf")
        plt.close()


def main_02():
    """ test with sparse matrices from 'https://sparse.tamu.edu' """

    # testing parameters
    iterations_per_matrix = 50
    tolerance = 1e-15
    t = -10**(-0)
    counter = np.zeros(1)


    package_runtimes, counters = {}, {}
    for spm_set in ['set_dziekonski']:
        package_runtimes[spm_set] = {
            'spm_info'  : [],
            'scipy'     : [],
            'numpy'     : {'expokit' : [], 'arnoldi' : [], 'lanczos' : []},
            'numba'     : {'expokit' : [], 'arnoldi' : [], 'lanczos' : []},
            'cupy'      : {'expokit' : [], 'arnoldi' : [], 'lanczos' : []}
        }
        counters[spm_set] = {'expokit' : [], 'arnoldi' : [], 'lanczos' : []}


        for spm in ['dielFilterV2real.mtx', 'dielFilterV3real.mtx', 'dielFilterV2clx.mtx', 'dielFilterV3clx.mtx']:
            print("    testing sparse matrix {} of set {}".format(spm, spm_set), "\n")


            # read sparse matrix
            path_to_spm = os.path.join(os.getcwd(), "spm_templates", spm_set, spm)
            row, col, data = [], [], []
            with open(path_to_spm, "r") as spm_file:
                lines = spm_file.readlines()
                spm_info = lines[59].split()
                matrix_size, nonzero_count = int(spm_info[0]), int(spm_info[2])
                if "real" in spm:
                    for line in lines[60:]:
                        entries = line.split()
                        row.append(int(entries[0]))
                        col.append(int(entries[1]))
                        data.append(float(entries[2]))
                    row, col = np.array(row, dtype=np.int32) - 1, np.array(col, dtype=np.int32) - 1
                    data = np.array(data, dtype=np.float64)
                else:
                    for line in lines[60:]:
                        entries = line.split()
                        row.append(int(entries[0]))
                        col.append(int(entries[1]))
                        if entries[3] == "0":
                            data.append(float(entries[2]) + 1j * 0.)
                        else:
                            data.append(0. + 1j * float(entries[2]))
                    row, col = np.array(row, dtype=np.int32) - 1, np.array(col, dtype=np.int32) - 1
                    data = np.array(data, dtype=np.complex128)

            matrix = csr_matrix((data, (row, col)), shape=(matrix_size, matrix_size))
            matrix += matrix.conj().transpose()
            for ind in [4173, 21902, 834]:
                row_, col_ = row[ind], col[ind]
            package_runtimes[spm_set]['spm_info'].append([matrix_size, nonzero_count])


            # build vector
            vector = np.random.ranf(matrix_size)
            if not "real" in spm:
                vector = vector + 1j * np.random.ranf(matrix_size)
            vector = vector / np.linalg.norm(vector)


            # use scipy (Pade approximation)
            #matrix_scipy = csc_matrix(matrix)
            #result_scipy = expm(t * matrix_scipy).dot(vector)
            result_scipy = 0
            function = lambda : expm(t * matrix_scipy).dot(vector)
            #runtime = timeit.timeit(function, number=1)
            runtime = 0
            package_runtimes[spm_set]['scipy'].append(runtime)
            print("SCIPY                               : {} seconds".format(runtime))


            # use expokit with numpy mat-vec product
            norm = onenormest(matrix)
            matvec_np = matvec_numpy(matrix, counter)
            m = 12
            if "real" in spm:
                result_scipy = dsexpv(vector, norm, m, t, matvec_np)
            else:
                result_scipy = zhexpv(vector, norm, m, t, matvec_np)
            counters[spm_set]['expokit'].append(counter[0])
            counter[0] = 0
            deviation = np.sqrt(np.sum(np.abs(result_scipy)**2))
            if "real" in spm:
                function = lambda : dsexpv(vector, norm, m, t, matvec_np)
            else:
                function = lambda : zhexpv(vector, norm, m, t, matvec_np)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['numpy']['expokit'].append(runtime)
            print("EXPOKIT (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use expokit with numba mat-vec product
            matvec_nu = matvec_numba(matrix)
            if "real" in spm:
                result = dsexpv(vector, norm, m, t, matvec_nu)
            else:
                result = zhexpv(vector, norm, m, t, matvec_nu)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            if "real" in spm:
                function = lambda : dsexpv(vector, norm, m, t, matvec_nu)
            else:
                function = lambda : zhexpv(vector, norm, m, t, matvec_nu)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['numba']['expokit'].append(runtime)
            print("EXPOKIT (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use expokit with cupy mat-vec product
            matvec_cp = matvec_cupy(matrix)
            if "real" in spm:
                result = dsexpv(vector, norm, m, t, matvec_cp)
            else:
                result = zhexpv(vector, norm, m, t, matvec_cp)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            if "real" in spm:
                function = lambda : dsexpv(vector, norm, m, t, matvec_cp)
            else:
                function = lambda : zhexpv(vector, norm, m, t, matvec_cp)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['cupy']['expokit'].append(runtime)
            print("EXPOKIT (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


            # use arnoldi with numpy mat-vec product
            result = expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
            counters[spm_set]['arnoldi'].append(counter[0])
            counter[0] = 0
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_arnoldi(vector, matrix, t, matvec_np, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['numpy']['arnoldi'].append(runtime)
            print("ARNOLDI (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use arnoldi with numba mat-vec product
            result = expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_arnoldi(vector, matrix, t, matvec_nu, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['numba']['arnoldi'].append(runtime)
            print("ARNOLDI (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use arnoldi with cupy mat-vec product
            result = expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_arnoldi(vector, matrix, t, matvec_cp, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            package_runtimes[spm_set]['cupy']['arnoldi'].append(runtime)
            print("ARNOLDI (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))


            # use lanczos with numpy mat-vec product
            result = expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
            counters[spm_set]['lanczos'].append(counter[0])
            counter[0] = 0
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_lanczos(vector, matrix, t, matvec_np, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            if np.isnan(deviation):
                runtime = 0
            package_runtimes[spm_set]['numpy']['lanczos'].append(runtime)
            print("LANCZOS (with numpy mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use lanczos with numba mat-vec product
            result = expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_lanczos(vector, matrix, t, matvec_nu, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            if np.isnan(deviation):
                runtime = 0
            package_runtimes[spm_set]['numba']['lanczos'].append(runtime)
            print("LANCZOS (with numba mat-vec product): {} seconds and {} deviation".format(runtime, deviation))

            # use lanczos with cupy mat-vec product
            result = expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
            deviation = np.sqrt(np.sum(np.abs(result_scipy - result)**2))
            function = lambda : expv_lanczos(vector, matrix, t, matvec_cp, tol=tolerance)
            runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
            if np.isnan(deviation):
                runtime = 0
            package_runtimes[spm_set]['cupy']['lanczos'].append(runtime)
            print("LANCZOS (with cupy  mat-vec product): {} seconds and {} deviation".format(runtime, deviation))
            print("\n")


    # plot the results
    plot_dir = "runtime_plots_t_{}".format(t)
    os.mkdir(plot_dir)
    colors, linestyles = ["k", "b", "g", "r"], ["-", "--", ":"]

    # one plot for each python package
    width = 0.25
    for spm_set in package_runtimes.keys():
        labels = ["({}, {})".format(*set_info) for set_info in package_runtimes[spm_set]['spm_info']]
        x = np.arange(len(labels))

        for i, items in enumerate(package_runtimes[spm_set].items()):
            pack_type, runtimes = items
            if pack_type == "spm_info":
                continue
            elif pack_type == "scipy":
                continue
            else:
                fig, ax = plt.subplots()
                plt.title("t = {} & {} mat-vec product".format(t, pack_type))
                for j, items_ in enumerate(runtimes.items()):
                    alg_type, runtimes_ = items_
                    rects = ax.bar(x + (j - 1)* width, runtimes_, width, label=alg_type)
                plt.xlabel("(matrix size (along one dimension), number of nonzero elements)")
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
                plt.ylabel("runtime (sec)")
                plt.yscale("log")
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "{}_{}_t{}.pdf".format(spm_set, pack_type, t)), format="pdf")
                plt.close()


    # plot matvec function call counts
    for spm_set in counters.keys():
        labels = ["({}, {})".format(*set_info) for set_info in package_runtimes[spm_set]['spm_info']]
        x = np.arange(len(labels))

        fig, ax = plt.subplots()
        plt.title("matvec call counts")
        for i, items in enumerate(counters[spm_set].items()):
            alg_type, call_counts = items
            rects = ax.bar(x + (i - 1)* width, call_counts, width, label=alg_type)
        plt.xlabel("matrix size (along one dimension)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.ylabel("matvec call count")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "matvec_call_count_{}_{}.pdf".format(spm_set,  t)), format="pdf")
        plt.close()


def main_03():
    """ test with sparse matrices from 'https://sparse.tamu.edu' """

    # testing parameters
    iterations_per_matrix = 20
    t = -10**(-2)
    counter = np.zeros(1)


    spm_set = 'set_dziekonski'
    spms = ['dielFilterV2real.mtx', 'dielFilterV3real.mtx', 'dielFilterV2clx.mtx', 'dielFilterV3clx.mtx']
    spm = spms[3]
    runtimes = {'numpy' : [], 'numba' : [], 'cupy' : []}
    counters, results = [], []


    print('  READING TEST MATRIX AND BUILDING TEST VECTOR')
    # read sparse matrix
    path_to_spm = os.path.join(os.getcwd(), "spm_templates", spm_set, spm)
    row, col, data = [], [], []
    with open(path_to_spm, "r") as spm_file:
        lines = spm_file.readlines()
        spm_info = lines[59].split()
        matrix_size, nonzero_count = int(spm_info[0]), int(spm_info[2])
        if "real" in spm:
            for line in lines[60:]:
                entries = line.split()
                row.append(int(entries[0]))
                col.append(int(entries[1]))
                data.append(float(entries[2]))
            row, col = np.array(row, dtype=np.int32) - 1, np.array(col, dtype=np.int32) - 1
            data = np.array(data, dtype=np.float64)
        else:
            for line in lines[60:]:
                entries = line.split()
                row.append(int(entries[0]))
                col.append(int(entries[1]))
                if entries[3] == "0":
                    data.append(float(entries[2]) + 1j * 0.)
                else:
                    data.append(0. + 1j * float(entries[2]))
            row, col = np.array(row, dtype=np.int32) - 1, np.array(col, dtype=np.int32) - 1
            data = np.array(data, dtype=np.complex128)

    matrix = csr_matrix((data, (row, col)), shape=(matrix_size, matrix_size))
    matrix += matrix.conj().transpose()

    # build vector
    vector = np.random.ranf(matrix_size)
    if not "real" in spm:
        vector = vector + 1j * np.random.ranf(matrix_size)
    vector = vector / np.linalg.norm(vector)


    print('  STARTING TEST')
    m_list = list(range(4, 18))
    for i, m in enumerate(m_list):

        print('    TESTING EXPOKIT FOR m = {}'.format(m))

        # use expokit with cupy mat-vec product
        norm = onenormest(matrix)
        matvec_cp = matvec_cupy(matrix)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_cp)
        else:
            result = zhexpv(vector, norm, m, t, matvec_cp)
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_cp)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_cp)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['cupy'].append(runtime)
        print("      cupy : {} seconds, {} l2-norm".format(runtime, np.linalg.norm(result)))

        # use expokit with numpy mat-vec product
        matvec_np = matvec_numpy(matrix, counter)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_np)
        else:
            result = zhexpv(vector, norm, m, t, matvec_np)
        results.append(result)
        counters.append(counter[0])
        counter[0] = 0
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_np)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_np)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['numpy'].append(runtime)
        print("      numpy: {} seconds".format(runtime))

        # use expokit with numba mat-vec product
        matvec_nu = matvec_numba(matrix)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_nu)
        else:
            result = zhexpv(vector, norm, m, t, matvec_nu)
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_nu)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_nu)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['numba'].append(runtime)
        print("      numba: {} seconds".format(runtime))


    # plot the results
    plot_dir = "runtime_plots_{}".format(spm)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    colors = ["b", "g", "r"]

    # one plot for all python packages
    plt.title('runtimes `{}`'.format(spm))
    for i, pack_type in enumerate(runtimes.keys()):
        plt.plot(m_list, runtimes[pack_type], color=colors[i], label=pack_type)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list), max(m_list)])
    plt.ylabel("runtimes (seconds)")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_runtimes_{}.pdf".format(t, spm)), format="pdf")
    plt.close()

    # one plot for matvec function call counts
    plt.title('matvec function call counts `{}`'.format(spm))
    plt.plot(m_list, counters)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list), max(m_list)])
    plt.ylabel("matvec function call count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_counts_{}.pdf".format(t, spm)), format="pdf")
    plt.close()

    # one plot for deviations
    deviations = [np.linalg.norm(results[i] - results[i + 1]) for i in range(len(results) - 1)]
    plt.title('deviations `{}`'.format(spm))
    plt.plot(m_list[1:], deviations)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list[1:]), max(m_list[1:])])
    plt.ylabel("deviation from previous result (l2-norm of diff.)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_deviations_{}.pdf".format(t, spm)), format="pdf")
    plt.close()


def main_04():
    """ test with sparse matrices from 'https://sparse.tamu.edu' """

    # testing parameters
    iterations_per_matrix = 20
    t = -10**(-5)
    counter = np.zeros(1)
    matrix_size = 10000
    spm = 'densemat3'


    runtimes = {'numpy' : [], 'numba' : [], 'cupy' : []}
    counters, results = [], []


    print('  BUILDING TEST MATRIX AND VECTOR')
    # build matrix
    matrix = random(int(matrix_size), int(matrix_size), density=1,  format="csr") + \
         1j * random(int(matrix_size), int(matrix_size), density=1,  format="csr")
    matrix += matrix.conj().transpose()
    matrix /= 2

    # build vector
    vector = np.random.ranf(matrix_size) + 1j * np.random.ranf(matrix_size)
    vector = vector / np.linalg.norm(vector)


    print('  STARTING TEST')
    m_list = list(range(4, 18))
    for i, m in enumerate(m_list):

        print('    TESTING EXPOKIT FOR m = {}'.format(m))

        # use expokit with cupy mat-vec product
        norm = onenormest(matrix)
        matvec_cp = matvec_cupy(matrix)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_cp)
        else:
            result = zhexpv(vector, norm, m, t, matvec_cp)
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_cp)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_cp)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['cupy'].append(runtime)
        print("      cupy : {} seconds, {} l2-norm".format(runtime, np.linalg.norm(result)))

        # use expokit with numpy mat-vec product
        matvec_np = matvec_numpy(matrix, counter)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_np)
        else:
            result = zhexpv(vector, norm, m, t, matvec_np)
        results.append(result)
        counters.append(counter[0])
        counter[0] = 0
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_np)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_np)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['numpy'].append(runtime)
        print("      numpy: {} seconds".format(runtime))

        # use expokit with numba mat-vec product
        matvec_nu = matvec_numba(matrix)
        if "real" in spm:
            result = dsexpv(vector, norm, m, t, matvec_nu)
        else:
            result = zhexpv(vector, norm, m, t, matvec_nu)
        if "real" in spm:
            function = lambda : dsexpv(vector, norm, m, t, matvec_nu)
        else:
            function = lambda : zhexpv(vector, norm, m, t, matvec_nu)
        runtime = timeit.timeit(function, number=iterations_per_matrix) / iterations_per_matrix
        runtimes['numba'].append(runtime)
        print("      numba: {} seconds".format(runtime))


    # plot the results
    plot_dir = "runtime_plots_{}".format(spm)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    colors = ["b", "g", "r"]

    # one plot for all python packages
    plt.title('runtimes `{}`'.format(spm))
    for i, pack_type in enumerate(runtimes.keys()):
        plt.plot(m_list, runtimes[pack_type], color=colors[i], label=pack_type)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list), max(m_list)])
    plt.ylabel("runtimes (seconds)")
    plt.yscale('log')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_runtimes_{}.pdf".format(t, spm)), format="pdf")
    plt.close()

    # one plot for matvec function call counts
    plt.title('matvec function call counts `{}`'.format(spm))
    plt.plot(m_list, counters)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list), max(m_list)])
    plt.ylabel("matvec function call count")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_counts_{}.pdf".format(t, spm)), format="pdf")
    plt.close()

    # one plot for deviations
    deviations = [np.linalg.norm(results[i] - results[i + 1]) for i in range(len(results) - 1)]
    plt.title('deviations `{}`'.format(spm))
    plt.plot(m_list[1:], deviations)
    plt.xlabel("basis size of Krylov subspace")
    plt.xlim([min(m_list[1:]), max(m_list[1:])])
    plt.ylabel("deviation from previous result (l2-norm of diff.)")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "t_{}_deviations_{}.pdf".format(t, spm)), format="pdf")
    plt.close()


if __name__ == "__main__":

    from scipy.sparse import random, csr_matrix, csc_matrix
    from scipy.sparse.linalg import onenormest
    import timeit, sys, os
    import matplotlib.pyplot as plt

    main_03()
