from richmol.rchm import old_to_new_richmol
from richmol.rchm import inspect_tensors, read_descr

######################################################################################
# example of conversion of richmol files from old (*.rchm) into new (*.h5) file format
######################################################################################

h5_file = "H2O_j0_j5.h5"
states_file = "../trove_rchm/energies_j0_j40_MARVEL_HITRAN.rchm"
dipole_file = "../trove_rchm/matelem_MU_j<j1>_j<j2>.rchm"

descr = "H2O/TROVE, energies are replaced with HITRAN values where available, assignment = k,tau,v1,v2,v3,rot_sym,vib_sym"
old_to_new_richmol(h5_file, states_file, dipole_file, replace=True, \
                   descr=descr, enr_units="1/cm", \
                   tens_descr="H2O dipole moment Eamon, matrix elements are truncated at J=5 and E=10000cm^-1", \
                   tens_units="Debye", tens_name="dipole")

# check hdf5 file
descr = read_descr(h5_file)
print(descr)

tens = inspect_tensors(h5_file)
for key in tens.keys():
    print("\n", key, tens[key])

