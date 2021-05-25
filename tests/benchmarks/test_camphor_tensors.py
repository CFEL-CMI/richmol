""" Read rotational energies and matrix elements for dipole and polarizability
      of R-camphor, calculated using older version of the program (cmirichmol),
      and compare them with the results of the same calculation done in Richmol.
"""




import numpy as np
from richmol.rot import solve, Solution, LabTensor, Molecule
from richmol.field import CarTens
from richmol.convert_units import MHz_to_invcm




# filter for reading old-format Richmol files
def filter(**kw):
    if 'J' in kw:
        return kw['J'] <= 20
    return True





if __name__ == "__main__":

    print(__doc__)



    # READ OLD-FORMAT RICHMOL FILES


    path = "tests/benchmarks/data/r-camphor_rchm_files/"

    # states file
    states_file = path + "camphor_energies_j0_j20.rchm"

    # template for generating names of matrix elements files
    dip_file = path + "camphor_matelem_mu_j<j1>_j<j2>.rchm"
    pol_file = path + "camphor_matelem_alpha_j<j1>_j<j2>.rchm"


    # load stationary states
    states = CarTens(states_file, bra=filter, ket=filter)

    # load dipole and polarizability tensors
    dip = CarTens(states_file, matelem=dip_file, bra=filter, ket=filter)
    pol = CarTens(states_file, matelem=pol_file, bra=filter, ket=filter)



    # CALCULATE THE SAME DATA USING RICHMOL


    camphor = Molecule()
    camphor.XYZ = (
        "angstrom",
        "O",      2.547204,    0.187936,   -0.213755,
        "C",      1.382858,   -0.147379,   -0.229486,
        "C",      0.230760,    0.488337,    0.565230,
        "C",      0.768352,   -1.287324,   -1.044279,
        "C",      0.563049,    1.864528,    1.124041,
        "C",     -0.716269,   -1.203805,   -0.624360,
        "C",     -0.929548,    0.325749,   -0.438982,
        "C",     -0.080929,   -0.594841,    1.638832,
        "C",     -0.791379,   -1.728570,    0.829268,
        "C",     -2.305990,    0.692768,    0.129924,
        "C",     -0.730586,    1.139634,   -1.733020,
        "H",      1.449798,    1.804649,    1.756791,
        "H",      0.781306,    2.571791,    0.321167,
        "H",     -0.263569,    2.255213,    1.719313,
        "H",     -1.413749,   -1.684160,   -1.316904,
        "H",      0.928638,   -1.106018,   -2.110152,
        "H",      1.245108,   -2.239900,   -0.799431,
        "H",     -1.816886,   -1.883799,    1.170885,
        "H",     -0.276292,   -2.687598,    0.915376,
        "H",      0.817893,   -0.939327,    2.156614,
        "H",     -0.738119,   -0.159990,    2.396232,
        "H",     -3.085409,    0.421803,   -0.586828,
        "H",     -2.371705,    1.769892,    0.297106,
        "H",     -2.531884,    0.195217,    1.071909,
        "H",     -0.890539,    2.201894,   -1.536852,
        "H",     -1.455250,    0.830868,   -2.487875,
        "H",      0.267696,    1.035608,   -2.160680
    )

    camphor.dip = [-1.21615, -0.30746, 0.01140]

    camphor.pol = [
        [115.80434,   0.58739,  -0.03276],
        [  0.58739, 112.28245,   1.36146],
        [ -0.03276,   1.36146, 108.47809]
    ]

    camphor.frame = "ipas"
    camphor.sym = "D2"

    sol = solve(camphor, Jmin=0, Jmax=20, verbose=True)

    states2 = LabTensor(camphor, sol)
    dip2 = LabTensor(camphor.dip, sol)
    pol2 = LabTensor(camphor.pol, sol)



    # COMPARE THE RESULTS


    maxdiff = {}

    # convert field-free H to full-matrix representation;
    #   do it for all Cartesian components
    mat = {
        cart : states.tomat(form='full', cart=cart, repres='csr_matrix')
            for cart in states.cart
    }
    mat2 = {
        cart : states2.tomat(form='full', cart=cart, repres='csr_matrix')
            for cart in states2.cart
    }

    # compute relative max difference
    for cart in mat.keys():
        maxdiff[cart] = np.max(np.abs(mat[cart] - mat2[cart]))


    # convert dipoles to full-matrix representation;
    #   do it for all Cartesian components
    mat = {
        cart : dip.tomat(form='full', cart=cart, repres='csr_matrix')
            for cart in dip.cart
    }
    mat2 = {
        cart : dip2.tomat(form='full', cart=cart, repres='csr_matrix')
            for cart in dip2.cart
    }

    # compute max difference
    for cart in mat.keys():
        maxdiff[cart] = np.max(np.abs(mat[cart] - mat2[cart]))


    # convert polarizabilities to full-matrix representation;
    #   do it for all Cartesian components
    mat = {
        cart : pol.tomat(form='full', repres='csr_matrix', cart=cart)
            for cart in pol.cart
    }
    mat2 = {
        cart : pol2.tomat(form='full', repres='csr_matrix', cart=cart)
            for cart in pol2.cart
    }

    for cart in mat.keys():
        maxdiff[cart] = np.max(np.abs(mat[cart] - mat2[cart]))

    print("Print maximal matrix element differences")
    for key, val in maxdiff.items():
        print(key, val)

