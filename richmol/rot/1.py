from richmol.rigrot import Molecule, solve, expval, wigexp, observ

# example of setting up molecule
h2s = Molecule()
h2s.XYZ = ()
# h2s.XYZ = "h2s.xyz"
oh2s.dip = [] # 
h2s.pol = [] # dip and pol attributes will be dynamically rotated if moleucle frame has changed
h2s.mat = [] # this will not be rotated
h2s.frame = "mat"
h2s.frame = "diag(pol)"
h2s.frame = "zyx,pas"
h2s.xyz_file = "h2s_new.xyz"
print(h2s.B) # print rotational constants Bx, By, Bz

sol = solve(h2s, J=[0,30], sym="D2") # solve eigenvalue problem
sol.enr
sol.coefs
sol.richmol_file = "h2s_richmol.h5"

ev = expval(h2s.dip, sol, sol) # compute expectation values of dipole moment

wig = wigexp(lambda theta, phi: np.cos(theta)**2, npoints=1000, jmax=100)
ev = expval(wigexp, sol, sol) # compute expectation values of Wigner-expanded function

ev.get(k1,m1,k2,m2)
ev.richmol_file = "h2s_richmol.h5"

spec = observ.dipspec(h2s.dip)
