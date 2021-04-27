import itertools
import numpy as np
from scipy.sparse import csr_matrix

mat = np.random.rand(6,6)

print(mat)


ind1 = [3,4,5]
ind2 = [1,2,3]

ind = np.array([comb for comb in itertools.product(ind1, ind2)])
rows = ind1#np.array(ind[:,0])
cols = ind2#np.array(ind[:,1])

print(rows)
print(cols)

mat2 = csr_matrix(mat)
mat3 = mat2[rows, :].tocsc()[:, cols].tocsr()

print(type(mat3))
