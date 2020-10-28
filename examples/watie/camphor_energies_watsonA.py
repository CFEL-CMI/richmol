from richmol.watie import RigidMolecule, symmetrize, SymtopBasis, \
        Jxx, Jyy, Jzz, Jxz, Jzx, Jyz, Jzy, Jxy, Jyx, JJ, Jp, Jm, \
        vellgt, settings
import numpy as np
import sys

####################################################################
# Example of calculation of rotational energies of camphor molecule
# using Watson Hamiltonian in A-reduction
####################################################################

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

# change molecule-fixed embedding to PAS frame
camphor.frame = "pas"

# The above structural information and molecular frame specification, i.e. XYZ and frame,
# are not really needed if Hamiltonian is build from the experimental constants

MHz = 1.0/vellgt*1e6
KHz = 1.0/vellgt*1e3

# parameters of Watson A-type Hamiltonian, from Supersonic expansion FTMW spectra, Kisiel, et al., PCCP 5, 820 (2003)
A, B, C  = (1446.968977*MHz, 1183.367110*MHz, 1097.101031*MHz)
DeltaJ = 0.0334804*KHz
DeltaJK = 0.083681*KHz
DeltaK = -0.06558*KHz
deltaJ = 0.0028637*KHz
deltaK = 0.024858*KHz

Jmax = 40

# don't use molecular symmetry

print("\nrotational energies (no symmetry)")
enr_all = []
for J in range(Jmax+1):
    bas = SymtopBasis(J)

    # Watson Hamiltonian in A reduction (use formula from pgopher documentation http://pgopher.chm.bris.ac.uk/Help/asymham.htm)

    # initialize various angular momentum operators
    Jx2 = Jxx(bas)
    Jy2 = Jyy(bas)
    Jz2 = Jzz(bas)
    J2 = JJ(bas)
    J4 = J2 * J2
    J6 = J2 * J4
    Jz2 = Jzz(bas)
    Jz4 = Jz2 * Jz2
    Jz6 = Jz2 * Jz4
    J2Jz2 = J2 * Jz2
    J4Jz2 = J2 * J2Jz2
    J2Jz4 = J2 * Jz4
    Jp2 = Jp() * Jp(bas)
    Jm2 = Jm() * Jm(bas)

    # build Hamiltonian
    H = C * Jx2 + B * Jy2 + A * Jz2 \
      - DeltaJ  * J4 - DeltaJK * J2Jz2 - DeltaK  * Jz4 \
      - 0.5 * DeltaJ * (J2*(Jp2+Jm2)+Jp2*J2+Jm2*J2) \
      - 0.5 * DeltaK * (Jz2*(Jp2+Jm2)+Jp2*Jz2+Jm2*Jz2)

    hmat = bas.overlap(H)                  # <psi|H|psi>
    enr, vec = np.linalg.eigh(hmat.real)   # eigenvalues and eigenvectors of <psi|H|psi>
    enr_all += [e for e in enr]
    bas2 = bas.rotate(krot=(vec.T, enr))   # rotate basis to the eigenvector representation
                                           #   and assign energies to new basis states
    # print states energies and assignments
    settings.assign_nprim = 2 # number of primitive states used for assignment (typically = 1)
    nprim = settings.assign_nprim
    for istate in range(bas2.nstates):
        print( J, bas2.sym[istate], " %12.6f"%bas2.enr[istate], \
                "  ".join(s+"=%s"%q for s,q, in zip(("| J","k","tau","abs(c)^2")*nprim, bas2.assign[istate])) )

# repeat previous calculation using D2 symmetry

print("\nrotational energies (D2 symmetry)")
enr_all_d2 = []
for J in range(Jmax+1):
    bas_d2 = symmetrize(SymtopBasis(J), sym="D2")
    for sym,bas in bas_d2.items():   # loops over basis sets for different irreps

        # initialize various angular momentum operators
        Jx2 = Jxx(bas)
        Jy2 = Jyy(bas)
        Jz2 = Jzz(bas)
        J2 = JJ(bas)
        J4 = J2 * J2
        J6 = J2 * J4
        Jz2 = Jzz(bas)
        Jz4 = Jz2 * Jz2
        Jz6 = Jz2 * Jz4
        J2Jz2 = J2 * Jz2
        J4Jz2 = J2 * J2Jz2
        J2Jz4 = J2 * Jz4
        Jp2 = Jp() * Jp(bas)
        Jm2 = Jm() * Jm(bas)

        # build Hamiltonian
        H = C * Jx2 + B * Jy2 + A * Jz2 \
          - DeltaJ  * J4 - DeltaJK * J2Jz2 - DeltaK  * Jz4 \
          - 0.5 * DeltaJ * (J2*(Jp2+Jm2)+Jp2*J2+Jm2*J2) \
          - 0.5 * DeltaK * (Jz2*(Jp2+Jm2)+Jp2*Jz2+Jm2*Jz2)

        hmat = bas.overlap(H)
        enr, vec = np.linalg.eigh(hmat.real)
        enr_all_d2 += [e for e in enr]
        bas2 = bas.rotate(krot=(vec.T, enr))
        bas2.sym = sym # assign symmetry to basis

        # print states energies and assignments
        settings.assign_nprim = 2
        nprim = settings.assign_nprim
        for istate in range(bas2.nstates):
            print( J, bas2.sym[istate], " %12.6f"%bas2.enr[istate], \
                    "  ".join(s+"=%s"%q for s,q, in zip(("| J","k","tau","abs(c)^2")*nprim, bas2.assign[istate])) )


# check if energies computed using different ways of setting up the Hamiltonian and basis agree with each other

tol = 1e-12
print(all(abs(x-y)<tol for x,y in zip(sorted(enr_all),sorted(enr_all_d2))) )

# compare with energies from pgopher

print("\nCompare with pgopher")
enr_pgopher = []
with open("examples/watie/camphor_watsonA_pgopher.txt", "r") as fl:
    lines = fl.readlines()
    for line in lines[2:]:
        w = line.split()
        enr_pgopher.append(float(w[3])*MHz)

print("\nrotational energies, computed - pgopher")
for ep,e in zip(sorted(enr_pgopher),sorted(enr_all)):
    print(ep,e,"%6.4e"%(ep-e))
