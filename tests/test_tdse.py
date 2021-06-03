import unittest
from richmol.field import CarTens
from richmol.convert_units import AUpol_x_Vm_to_invcm
from richmol.tdse import TDSE
import numpy as np
import matplotlib.pyplot as plt


class testTDSE(unittest.TestCase):
    """
    """

    ### Test OCS alignment: reproduce chronological sequence of occupation
    ##    robabilities using Lanczos method for matrix exponential
    def test_alignment_ocs_lanczos(self):

        #  read reference chronological sequence of occupation probabilities
        path = 'tests/etc/alignment_ocs/'
        fname_ref_pop = path + 'pop_reference.txt'
        J_list = [0, 2, 4, 6, 8, 10, 12]
        ref_occu_probs = []
        with open(fname_ref_pop, 'r') as f:
            next(f)
            for line in f:
                split_line = line.split()
                probs = []
                for J_ind in range(len(J_list)):
                    if 6 * (J_ind + 1) < len(split_line):
                        probs.append(float(split_line[2 + 6 * J_ind]))
                    else:
                        probs.append(0.0)
                ref_occu_probs.append(probs)
        ref_occu_probs = np.array(ref_occu_probs)

        # tensor
        def filt(**kwargs):
            J_pass, M_pass = True, True
            if 'J' in kwargs:
                J_pass = kwargs['J'] <= 30 and kwargs['J'] % 2 == 0
            if 'm' in kwargs:
                M_pass = kwargs['m'] == ' 0.0'
            return J_pass and M_pass
        filename = path + 'matelem/ocs_energies_j0_j30.rchm'
        matelem = path + 'matelem/ocs_matelem_alpha_j<j1>_j<j2>.rchm'
        H0 = CarTens(filename, bra=filt, ket=filt) * [0, 0, 1]
        Hbar = CarTens(filename, matelem, bra=filt, ket=filt) \
            * (-0.5) * AUpol_x_Vm_to_invcm()

        # field (V/m)
        fname_field = path + 'field.txt'
        with  open(fname_field, 'r') as f:
            field = [[0, 0, 1e2 * float(line.split()[3])] for line in f]

        # compute chronological sequence of occupation probabilities
        tdse = TDSE()
        tdse.tstart = 0
        tdse.tend = 10
        tdse.dt = 0.01
        tdse.time_units = 'ps'
        tdse.energy_units = 'invcm'
        occu_probs, vecs = [], None
        for ind, _ in enumerate(tdse.time_grid()):
            Hbar.field(field[ind])
            vecs, t = tdse.update(Hbar, H0=H0, vecs=vecs, matvec_lib='scipy')
            if ind % 10 == 0:
                occu_probs.append(
                   [ round(t - 0.01, 2),
                     *[round(abs(vecs[0][int(J / 2)])**2, 4) for J in J_list] ]
                )
                #print('    result: ', occu_probs[-1],
                #      '    reference: ', ref_occu_probs[int(ind / 10)])
        occu_probs = np.array(occu_probs)

        # write chronological sequence of occupation probabilities
        np.savetxt(
            path + 'pop_lanczos.txt',
            occu_probs,
            fmt = '  %6.1f' + 7 * '    %6.4f',
            header = 't (ps)' + \
                ''.join([f"    J = {'{:2.0f}'.format(J)}"for J in J_list])
        ) 

        # plot chronological sequence of occupation probability
        ref_occu_probs = ref_occu_probs[: occu_probs.shape[0]]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for J_ind in range(len(J_list)):
            c = colors[J_ind % len(J_list)]
            plt.plot(
                occu_probs.T[0],
                occu_probs.T[J_ind + 1],
                c = colors[J_ind % len(J_list)],
                label = f"result (J = {J_list[J_ind]})"
            )
            plt.plot(
                occu_probs.T[0],
                (-1) * ref_occu_probs.T[J_ind],
                c = colors[J_ind % len(J_list)],
                label = f"reference (J = {J_list[J_ind]})",
                linestyle = 'dashed'
            )
        plt.title('time-evolution of occupation probabilities')
        plt.xlabel('time (ps)')
        plt.ylabel('occupation probability')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(path + 'occu_probs_ref_lanczos.png', format='png')
        plt.close()

        # check chronological sequence of occupation probabilites against reference
        occu_probs_devi = abs(occu_probs[:, 1:] - ref_occu_probs)
        assert (np.amax(occu_probs_devi) <= 0.001), \
            f"deviation from reference above tolerance registered"


if __name__ == "__main__":

    unittest.main()
