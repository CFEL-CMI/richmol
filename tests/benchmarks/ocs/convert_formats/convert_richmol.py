from richmol.rchm import old_to_new_richmol
from richmol.rchm import inspect_tensors, read_descr

######################################################################################
# example of conversion of richmol files from old (*.rchm) into new (*.h5) file format
######################################################################################

h5_file = "OCS_j0_j30.h5"
states_file = "../cmi-richmol_watie/ocs_energies_j0_j30.rchm"
dipole_file = "../cmi-richmol_watie/ocs_matelem_mu_j<j1>_j<j2>.rchm"
polariz_file = "../cmi-richmol_watie/ocs_matelem_alpha_j<j1>_j<j2>.rchm"

old_to_new_richmol(h5_file, states_file, dipole_file, replace=True, \
                   descr="converted from old richmol format", enr_units="1/cm", \
                   tens_descr="calculated CCSD(T)/ACVQZ", tens_units="a.u. of dipole moment", \
                   tens_name="dipole")

old_to_new_richmol(h5_file, states_file, polariz_file, replace=False, \
                   tens_descr="calculated CCSD(T)/ACVQZ", tens_units="a.u. of polarizability")

# check hdf5 file
descr = read_descr(h5_file)
print(descr)

tens = inspect_tensors(h5_file)
for key in tens.keys():
    print("\n", key, tens[key])

