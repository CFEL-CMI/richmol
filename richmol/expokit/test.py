import expokit
import numpy as np
from scipy.sparse.linalg import LinearOperator, onenormest, aslinearoperator
from scipy.sparse import random
import sys


def matvec(v):
    print('call')
    print(v)
    res = mat.dot(v)
    print(res)
    return res

#print(expokit.dsexpv.__doc__)
#sys.exit()

n = 10
m = 5

vec = np.random.ranf(n)
vec = vec / np.linalg.norm(vec)
mat = random(n, n, format="csr")
mat = (mat.T + mat) / 2
mat_norm = onenormest(mat)
M = aslinearoperator(mat)

workspace = np.zeros(n*(m+2)+5*(m+2)*(m+2)+7, dtype=np.float64)
iworkspace = np.zeros(m+3, dtype=np.int32)
t = 1.0
tol = 0.0

print(vec)
u,iflag0 = expokit.dsexpv(m, t, vec, tol, mat_norm, workspace, iworkspace, M.matvec, 0)
