from richmol.watie import RigidMolecule, SymtopBasis, vellgt, Jxx, Jyy, Jzz, settings, CartTensor
import numpy as np
import sys


####################################################################################################
# Example of a calculation of Richmol energy and matrix elements files for dipole moment and polarizability
# Rigid-rotor Hamiltonian based on experimental rotational constants
# No symmetry employed
####################################################################################################


camphor = RigidMolecule()

# S-Camphor structure from Supersonic expansion FTMW spectra, Kisiel, et al., PCCP 5, 820 (2003)
camphor.XYZ = ("angstrom", \
        "O",     -2.547204,    0.187936,   -0.213755, \
        "C",     -1.382858,   -0.147379,   -0.229486, \
        "C",     -0.230760,    0.488337,    0.565230, \
        "C",     -0.768352,   -1.287324,   -1.044279, \
        "C",     -0.563049,    1.864528,    1.124041, \
        "C",      0.716269,   -1.203805,   -0.624360, \
        "C",      0.929548,    0.325749,   -0.438982, \
        "C",      0.080929,   -0.594841,    1.638832, \
        "C",      0.791379,   -1.728570,    0.829268, \
        "C",      2.305990,    0.692768,    0.129924, \
        "C",      0.730586,    1.139634,   -1.733020, \
        "H",     -1.449798,    1.804649,    1.756791, \
        "H",     -0.781306,    2.571791,    0.321167, \
        "H",      0.263569,    2.255213,    1.719313, \
        "H",      1.413749,   -1.684160,   -1.316904, \
        "H",     -0.928638,   -1.106018,   -2.110152, \
        "H",     -1.245108,   -2.239900,   -0.799431, \
        "H",      1.816886,   -1.883799,    1.170885, \
        "H",      0.276292,   -2.687598,    0.915376, \
        "H",     -0.817893,   -0.939327,    2.156614, \
        "H",      0.738119,   -0.159990,    2.396232, \
        "H",      3.085409,    0.421803,   -0.586828, \
        "H",      2.371705,    1.769892,    0.297106, \
        "H",      2.531884,    0.195217,    1.071909, \
        "H",      0.890539,    2.201894,   -1.536852, \
        "H",      1.455250,    0.830868,   -2.487875, \
        "H",     -0.267696,    1.035608,   -2.160680)

# dipole moment and polarizability (in atomic units) computed using B3LYP/def2-TZVPP at the above geometry
camphor.tensor = ("dipole moment", [1.21615, -0.30746, 0.01140])
camphor.tensor = ("polarizability", [[115.80434, -0.58739, 0.03276], \
                                     [-0.58739, 112.28245, 1.36146], \
                                     [0.03276, 1.36146, 108.47809]] )

# change molecule-fixed embedding to the PAS frame
camphor.frame = "pas"

# rotational constants and asymmetry parameter calculated from the above Cartesian geometry input
Bx, By, Bz = camphor.B
kappa = camphor.kappa
print(Bx, By, Bz)  # 0.04826693384984668 0.03947392003309187 0.03659630461693979
print(kappa)       # -0.5068619936895491

# since kappa<0 swap axes in the PAS frame (i.e. abc -> bca), as a matter of the Ir convention for near prolate-top
camphor.frame = "yzx"

# print again rotational constants
Bx, By, Bz = camphor.B
print(Bx, By, Bz)  # 0.03947392003309187 0.03659630461693979 0.04826693384984668

# experimental rotational constants, from Supersonic expansion FTMW spectra, Kisiel, et al., PCCP 5, 820 (2003)
MHz = 1.0/vellgt*1e6
A, B, C  = (1446.968977*MHz, 1183.367110*MHz, 1097.101031*MHz)
print(B, C, A)    # 0.039472877933440204 0.03659535127464748 0.04826568975928007

# replace computed constants with experimental once
Bx, By, Bz = B, C, A

# we can print polarizability and dipole moment in new frame "yzx,pas"
print("polarizability:\n", camphor.tensor["polarizability"])
print("dipole moment:\n", camphor.tensor["dipole moment"])
# print inertia tensor (must be diagonal in the PAS frame)
print("inertia tensor:\n", camphor.imom())

# compute rotational energies for J = 0..20

Jmax = 20

settings.assign_nprim = 2 # number of primitive functions used for state assignment
settings.assign_ndig_c2 = 6 # number of digits printed for the assignment coefficient

wavefunc = {}
for J in range(Jmax+1):
    bas = SymtopBasis(J)                   # initialize basis of symmetric-top functions |psi>
    Jx2 = Jxx(bas)                         # Jx^2|psi>
    Jy2 = Jyy(bas)                         # Jy^2|psi>
    Jz2 = Jzz(bas)                         # Jz^2|psi>
    H = Bx * Jx2 + By * Jy2 + Bz * Jz2     # H|psi> = Bx*Jx^2|psi> + By*Jy^2|psi> + Bz*Jz^2|psi>
    hmat = bas.overlap(H)                  # <psi|H|psi>
    enr, vec = np.linalg.eigh(hmat.real)   # eigenvalues and eigenvectors of <psi|H|psi>
    wavefunc[J] = bas.rotate(krot=(vec.T, enr))   # rotate basis to the eigenvector representation
                                                  #   and assign energies to new basis states

# store rotational energies in Richmol energy file
for J in range(Jmax+1):
    wavefunc[J].store_richmol("camphor_energies_j0_j"+str(Jmax)+".rchm")

mu = CartTensor(camphor.tensor["dipole moment"])
alpha = CartTensor(camphor.tensor["polarizability"])

for J1 in range(Jmax+1):
    for J2 in range(J1,Jmax+1):
        if abs(J1-J2)>2: continue
        mu.store_richmol(wavefunc[J2], wavefunc[J1], thresh=1e-12, name="mu", fname="camphor_dip")
        alpha.store_richmol(wavefunc[J2], wavefunc[J1], thresh=1e-12, name="alpha", fname="camphor_pol")

