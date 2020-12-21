Rotational energies, dipole moment and polarizability matrix elements, as well as Stark-effect
and laser alignment simulations for OCS molecule, computed with different programs.
====================================================================================================

cmi-richmol_watie: rotational energies and matrix elements computed using watie module from cmi-richmol
                   package (https://stash.desy.de/scm/cmistark/cmi-richmol.git).

watie: rotational energies and matrix elements computed using watie module from the current package.
       The data is stored in two formats, the old formatted text files (same as in cmi-richmol_watie)
       and the new hdf5 file format.

extfield: simulations of Stark effect and alignment dynamics using extfield module from the current package.

richmol2: simulations of Stark effect and alignment dynamics using old richmol package
          (https://stash.desy.de/scm/cmistark/richmol.git).

convert_formats: old-format richmol files in cmi-richmol_watie are converted into new hdf5 file
                 format, the results are tested in simulations of Stark effect and alignment (see extfield)

Results of calculations 'richmol2 + cmi-richmol_watie' must agree with 'extfield + watie'.
In addition, to test format conversion function (from old *.rchm to new *.h5 format),
results of 'extfield + convert_formats' must agree with 'extfield + watie'.
