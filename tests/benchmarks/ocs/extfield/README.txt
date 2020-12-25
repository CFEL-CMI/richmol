Stark energies and field-dressed transition linestrengths for the following field configurations:
Fx, Fy, Fz = 0, 0, 100 kV/cm : run stark_fz.py, output stark_energies_fz10000000.txt and stark_transitions_fz10000000.txt
Fx, Fy, Fz = 0, 100, 0 kV/cm : run stark_fy.py, output stark_energies_fy10000000.txt and stark_transitions_fy10000000.txt
Fx, Fy, Fz = 100, 0, 0 kV/cm : run stark_fx.py, output stark_energies_fx10000000.txt and stark_transitions_fx10000000.txt
Fx, Fy, Fz = 0, 0, 100 kV/cm : run stark_fz_diag_m.py, output stark_energies_fz10000000_diag_m.txt and stark_transitions_fz10000000_diag_m.txt

Run 'python3 plot_compare.py stark_transitions*.txt stark_transitions*.txt' to compare results obtained with
different field configurations.

Run 'python3 plot_compare_with_richmol2.py stark_transitions*.txt' to compare results obtained with extfield
module and older richmol2 program.
