Here we test rovibrational energies and field-dressed transition linestrengths of water molecule
placed in static electric field of 10 kV/cm.

The initial richmol energy and dipole matrix elements files are computed with TROVE and stored
using old formatted text output in 'trove_rchm' folder. The matrix elements are truncated for
rovibrational states with J=0..5 and energies below 10000 cm^-1. The file with energies contains
computed energies substituted with HITRAN values where available (courtesy Sergei Yurchenko).

The rovibrational energies of water in static electric field, together with field-dressed transition
linestrengths are computed using older richmol2 in 'richmol2' folder.

The old-format richmol files in 'trove_rchm' are converted into a new hdf5 file format
in 'convert_rchm_to_h5' folder.

The rovibrational energies of water in static electric field, together with field-dressed transition
linestrengths are computed using current extfield module in 'extfield' folder.
