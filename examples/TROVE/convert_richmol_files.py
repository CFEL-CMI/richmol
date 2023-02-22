"""Example of how to convert '*.rchm' files produced by TROVE or HyFor
into richmol hdf5 file format.
"""
from richmol.trove import CarTensTrove


def correctForSignOfK1Tensor(data):
    """Changes sign of spin-rotation K(1) tensor when it is obtained,
    for example, for (J1, J2) pair of angular momentum quanta
    by transposing data for (J2, J1) pair.
    This is needed because in TROVE only matrix elements with J1 <= J2 are stored,
    while we need both (J1, J2) and (J2, J1) pairs for richmol hdf5.
    """
    print("Correct sign in spin-rotation K(1) matrix")
    for jpair in data.kmat.keys():
        j1, j2 = jpair
        if j2 > j1:
            sign = -1
        else:
            sign = 1
        for sympair in data.kmat[jpair].keys():
            data.kmat[jpair][sympair][1] *= sign
    return data


if __name__ == "__main__":

    # output richmol hdf5 file name
    jmax = 60
    richmol_file = f"h2s_p36_j{jmax}_rovib.h5"

    # set True to convert wave function coefficients as well
    # these are used for plotting rotational densities
    store_coefs = False

    # dictionary containing names of operators (as keys) with corresponding TROVE
    # matrix elements' file names along with a short description (e.g., units, details
    # of calculations, references, etc)
    # key = dataset name, value = (path to *.rchm files, comment string)
    new_data = {
        "h0" : (f"energies_j0_j{jmax}.rchm", #f"coefficients_j0_j{jmax}.rchm",
                "Authors: Andrey Yachmenev (CFEL/DESY). " + \
                # "Richmol database for H2S molecule (includes wavefunction coefficients for density plots). " + \
                "Richmol database for H2S molecule. " + \
                "TROVE variationally calculated rovibrational energies and matrix elements " + \
                "of spin-rotation, spin-spin, and electric dipole moment operators. " + \
                "The vibrational basis is truncated at polyad P=36, details of " + \
                "calculations can be found in https://doi.org/10.1093/mnras/stw1133."),
        "dipole" : ("dipole/matelem_MU_j<j1>_j<j2>.rchm",
                    "rovibrational matrix elements of electric dipole moment in Debye, " + \
                    "for details, see https://doi.org/10.1093/mnras/stw1133."),
        "spin-rot H1" : ("SR1/matelem_SR_j<j1>_j<j2>.rchm",
                         "rovibrational matrix elements of spin-rotation tensor for H1 " + \
                         "in kHz, surface: CCSD(T)/aug-cc-pCVT(+d)Z using CFOUR "),
        "spin-rot H2" : ("SR2/matelem_SR_j<j1>_j<j2>.rchm",
                         "rovibrational matrix elements of spin-rotation tensor for H2 " + \
                         "in kHz, surface: CCSD(T)/aug-cc-pCVT(+d)Z using CFOUR "),
        "spin-spin H1-H2" : ("SS/matelem_SR_j<j1>_j<j2>.rchm",
                             "spin-spin (magnetic dipole-dipole interaction) tensor for H1-H2 in kHz"),
    }

    for key in new_data.keys():
        print(f"read dataset '{key}' from '{new_data[key][0]}'")
        if key == "h0":
            try:
                states_file, coefs_file, descr = new_data[key]
            except ValueError:
                states_file, descr = new_data[key]
                coefs_file = None
            if store_coefs:
                data = CarTensTrove(states_file, coefs=coefs_file)
            else:
                data = CarTensTrove(states_file)
        else:
            data = CarTensTrove(states_file, new_data[key][0])
        if key in ("spin-rot H1", "spin-rot H2"):
            data = correctForSignOfK1Tensor(data)

        print(f"add dataset '{key}' to '{richmol_file}'")
        data.store(richmol_file, name=key, comment=new_data[key][1], replace=True)

