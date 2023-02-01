""" Read time-evolution of state coefficients for OCS alignment, calculated
      using older version of the program (cmirichmol), and compare them with
      the results of the same calculation done in Richmol
"""




from richmol.trove import CarTensTrove
from richmol.convert_units import AUpol_x_Vm_to_invcm
from richmol.tdse import TDSE
import numpy as np
import matplotlib.pyplot as plt




def test_alignment_ocs():

    # tolerance
    occu_prob_tol = 0.001

    # end time (ps); time step of propagation (ps)
    t_end = 250
    dt = 0.01

    # quantum numbers J to use for reference check
    J_list = [0, 2, 4, 6, 8, 10, 12]

    # read reference chronological sequence of occupation probabilities
    path = '/home/yachmena/richmol/tests/benchmarks/data/alignment_ocs/'
    fname_ref_pop = path + 'pop_reference.txt'
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
    h0 = CarTensTrove(filename, bra=filt, ket=filt)
    alpha = CarTensTrove(filename, matelem, bra=filt, ket=filt)

    # field (V/m)
    fname_field = path + 'field.txt'
    with  open(fname_field, 'r') as f:
        field = [[0, 0, 1e2 * float(line.split()[3])] for line in f]

    tdse = TDSE(t_start=0, t_end=t_end, dt=dt, t_units="ps", enr_units="invcm")
    vec = tdse.init_state(h0, temp=0)

    # compute chronological sequence of occupation probabilities
    occu_probs = []
    for i, t in enumerate(tdse.time_grid()):
        print('  t = {:6.2f} ps'.format(t), end='\r')
        ham = -0.5 * AUpol_x_Vm_to_invcm() * alpha * field[i]
        vec, t_ = tdse.update(ham, H0=h0, vecs=vec, matvec_lib='scipy')
        if i % 10 == 0:
            occu_probs.append(
                [round(t, 1), *[round(abs(elem)**2, 4) for _, elem in zip(J_list, vec[0, :])]]
            )
            print('    result: ', occu_probs[-1],
                  '    reference: ', ref_occu_probs[int(i / 10)])
    occu_probs = np.array(occu_probs)

    # write chronological sequence of occupation probabilities
    np.savetxt(
        path + 'pop_lanczos.txt',
        occu_probs,
        fmt = '   %5.1f' + 7 * '    %6.4f',
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
    occu_probs_devi = abs(
        occu_probs[:, 1:] - ref_occu_probs
    )
    assert (np.amax(occu_probs_devi) <= occu_prob_tol), \
        f"deviation from reference above tolerance registered"


if __name__ == "__main__":

    test_alignment_ocs()
