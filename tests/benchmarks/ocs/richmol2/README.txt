Stark energies and field-dressed linestrengths calculations using older richmol2 program with old-format
richmol matrix elements files.
Fx, Fy, Fz = 0, 0, 100 kV/cm : run stark_dc100kv_z.inp, output stark_energies_100000.0_0.0_z and mu_me_z
Fx, Fy, Fz = 0, 100, 0 kV/cm : run stark_dc100kv_y.inp, output stark_energies_100000.0_0.0_y and mu_me_y
Fx, Fy, Fz = 100, 0, 0 kV/cm : run stark_dc100kv_x.inp, output stark_energies_100000.0_0.0_x and mu_me_x

Files mu_me* contain transition linestrengths (without Gns factors) for different transition frequencies.
Run 'python3 plot_compare.py mu_me* mu_me*' to compare results obtained with different field configurations.
