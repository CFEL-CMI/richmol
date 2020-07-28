from richmol.watie import RigidMolecule, symmetrize, SymtopBasis, Jxx, Jyy, Jzz, Jxz, Jzx, Jyz, \
                          Jzy, Jxy, Jyx, CartTensor
import numpy as np
import sys


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

# dipole moment and polarizability (in atomic units) computed using B3LYP/def2-TZVPP at experimental geometry
camphor.tensor = ("dipole moment", [1.21615, -0.30746, 0.01140])
camphor.tensor = ("polarizability", [[115.80434, -0.58739, 0.03276], \
                                     [-0.58739, 112.28245, 1.36146], \
                                     [0.03276, 1.36146, 108.47809]] )

# change molecule-fixed embedding to PAS frame
camphor.frame = "pas"

# compute rotational energies for J=0..20
# use rigid-rotor Hamiltonian constructed from rotational constants

Jmax = 20

# don't employ symmetry
print("rotational energies (no symmetry)")
print(" energy [cm^-1]   J    k  tau  |c|^2")
enr_all = []
for J in range(Jmax+1):
    bas = SymtopBasis(J)              # initialize basis of symmetric-top functions
    Jx2 = Jxx(bas)                    # Jx^2|psi>
    Jy2 = Jyy(bas)                    # Jy^2|psi>
    Jz2 = Jzz(bas)                    # Jz^2|psi>
    A, B, C = camphor.ABC
    H = B * Jx2 + C * Jy2 + A * Jz2   # H|psi> = A*Jx^2|psi> + B*Jy^2|psi> + C*Jz^2|psi>
    hmat, _ = bas.overlap(H)             # <psi|H|psi>
    enr, vec = np.linalg.eigh(hmat)   # eigenvalues and eigenvectors of <psi|H|psi>
    enr_all += [e for e in enr]
    # print energies and assignments in the order:
    #   energy (cm^-1) J k tau |c|^2 | J' k' tau' |c'|^2 an so on for 'n' largest contributions
    # for e,v in zip(enr,vec.T):
    #     n = 1           # number of largest contributions (typically n=1)
    #     ind = (-abs(v)).argsort()[:n]
    #     c2 = np.abs(v[ind])**2
    #     print(" %14.4f"%e + " | ".join( " ".join(" %3i"%jkt for jkt in bas.jkt[ind,:][i]) \
    #                                    + " %6.3f"%c2[i] for i in range(n) ) )
print(sorted(enr_all))



