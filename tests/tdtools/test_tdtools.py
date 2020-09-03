import unittest
import copy
from richmol.tdtools import Psi, Etensor



class testEtensor(unittest.TestCase):
    """
    """
    # tolerance for computed occupation probabilities
    occu_prob_tol = 0.1

    # time (ps) to compute occupation probabilities up to
    t_end = 10


    # Test OCS alignment: reproduce chronological sequence of occupation
    #                     probabilities using Taylor method for matrix exponential
    def test_alignment_ocs_taylor(self):

        # matrix exponential method
        meth = "Taylor"

        # list of quantum numbers F to use for reference check
        f_list = [0, 2, 4, 6, 8, 10, 12]

        # time step of propagation (ps)
        dt = 0.01

        # read basis
        fname_enr = 'etc/ocs_alignment/matelem/ocs_energies_j0_j30.rchm'
        psi = Psi(fname_enr, fmin=0, fmax=30, mmin=0, mmax=0, dm=2, df=2)

        # initalize wavepacket (f, m, id, ideg, coef)
        psi.j_m_id = (0, 0, 1, 1, 1.0 + 0.0j)

        # read tensor(s)
        fname_tens = 'etc/ocs_alignment/matelem/ocs_matelem_alpha_j<j1>_j<j2>.rchm'
        alpha = Etensor(fname_tens, psi)

        # read field (V/m)
        fname_field = 'etc/ocs_alignment/field.txt'
        f_field = open(fname_field,'r')
        field = [[0, 0, 100 * float(line.split()[3])] for line in f_field]
        f_field.close()


        # compute chronological sequence of occupation probabilities
        occu_probs = []
        for t_ind in range(int(self.t_end / dt)):
            print('t = ' + '{:03.2f}'.format(t_ind * dt) + ' ps')
            H = -0.5 * alpha * field[t_ind] # NOTE: only accounts for perturbation Hamiltonian
            psi = H.U(dt, psi, method=meth)
            if int(t_ind % 10) == 0:
                occu_probs.append([round(abs(psi.coefs[f][0])**2, 4) for f in f_list])


        # write chronological sequence of occupation probabilities
        fname_occu_probs = 'etc/ocs_alignment/' + meth + '_pop.txt'
        f_occu_probs = open(fname_occu_probs, 'w')
        f_occu_probs.write('T\M,F    0,0       0,2       0,4       0,6       0,8       0,10      0,12\n')
        for t_ind in range(len(occu_probs)):
            f_occu_probs.write('{:05.1f}'.format(t_ind * 10 * dt))
            for prob in occu_probs[t_ind]:
                f_occu_probs.write('    ' + '{:.4f}'.format(prob))
            f_occu_probs.write('\n')
        f_occu_probs.close()


        # read reference chronological sequence of occupation probabilities
        ref_occu_probs = []
        fname_ref_pop = 'etc/ocs_alignment/reference_pop.txt'
        f_ref_pop = open(fname_ref_pop, 'r')
        next(f_ref_pop)
        for line in f_ref_pop:
            split_line = line.split()
            probs = []
            for f_ind in range(len(f_list)):
                if 6 * (f_ind + 1) < len(split_line):
                    probs.append(float(split_line[2 + 6 * f_ind]))
                else:
                    probs.append(0)
            ref_occu_probs.append(probs)
        f_ref_pop.close()


        # check chronological sequence of occupation probabilites against reference
        for t_ind in range(len(occu_probs)):
            for f_ind in range(len(f_list)):
                occu_prob_devi = abs(occu_probs[t_ind][f_ind] - ref_occu_probs[t_ind][f_ind])
                if not occu_prob_devi <= self.occu_prob_tol:
                    print(str(t_ind * 10 * dt) + str(f_ind))
                self.assertTrue(occu_prob_devi <= self.occu_prob_tol)


    # Test OCS alignment: reproduce chronological sequence of occupation
    #                     probabilities using Arnoldi method for matrix exponential
    def test_alignment_ocs_arnoldi(self):

        # matrix exponential method
        meth = "Arnoldi"

        # list of quantum numbers F to use for reference check
        f_list = [0, 2, 4, 6, 8, 10, 12]

        # time step of propagation (ps)
        dt = 0.01

        # read basis
        fname_enr = 'etc/ocs_alignment/matelem/ocs_energies_j0_j30.rchm'
        psi = Psi(fname_enr, fmin=0, fmax=30, mmin=0, mmax=0, dm=2, df=2)

        # initalize wavepacket (f, m, id, ideg, coef)
        psi.j_m_id = (0, 0, 1, 1, 1.0 + 0.0j)

        # read tensor(s)
        fname_tens = 'etc/ocs_alignment/matelem/ocs_matelem_alpha_j<j1>_j<j2>.rchm'
        alpha = Etensor(fname_tens, psi)

        # read field (V/m)
        fname_field = 'etc/ocs_alignment/field.txt'
        f_field = open(fname_field,'r')
        field = [[0, 0, 100 * float(line.split()[3])] for line in f_field]
        f_field.close()


        # compute chronological sequence of occupation probabilities
        occu_probs = []
        for t_ind in range(int(self.t_end / dt)):
            print('t = ' + '{:03.2f}'.format(t_ind * dt) + ' ps')
            H = -0.5 * alpha * field[t_ind] # NOTE: only accounts for perturbation Hamiltonian
            psi = H.U(dt, psi, method=meth)
            if int(t_ind % 10) == 0:
                occu_probs.append([round(abs(psi.coefs[f][0])**2, 4) for f in f_list])


        # write chronological sequence of occupation probabilities
        fname_occu_probs = 'etc/ocs_alignment/' + meth + '_pop.txt'
        f_occu_probs = open(fname_occu_probs, 'w')
        f_occu_probs.write('T\M,F    0,0       0,2       0,4       0,6       0,8       0,10      0,12\n')
        for t_ind in range(len(occu_probs)):
            f_occu_probs.write('{:05.1f}'.format(t_ind * 10 * dt))
            for prob in occu_probs[t_ind]:
                f_occu_probs.write('    ' + '{:.4f}'.format(prob))
            f_occu_probs.write('\n')
        f_occu_probs.close()


        # read reference chronological sequence of occupation probabilities
        ref_occu_probs = []
        fname_ref_pop = 'etc/ocs_alignment/reference_pop.txt'
        f_ref_pop = open(fname_ref_pop, 'r')
        next(f_ref_pop)
        for line in f_ref_pop:
            split_line = line.split()
            probs = []
            for f_ind in range(len(f_list)):
                if 6 * (f_ind + 1) < len(split_line):
                    probs.append(float(split_line[2 + 6 * f_ind]))
                else:
                    probs.append(0)
            ref_occu_probs.append(probs)
        f_ref_pop.close()


        # check chronological sequence of occupation probabilites against reference
        for t_ind in range(len(occu_probs)):
            for f_ind in range(len(f_list)):
                occu_prob_devi = abs(occu_probs[t_ind][f_ind] - ref_occu_probs[t_ind][f_ind])
                if not occu_prob_devi <= self.occu_prob_tol:
                    print(str(t_ind * 10 * dt) + str(f_ind))
                self.assertTrue(occu_prob_devi <= self.occu_prob_tol)


    # Test OCS alignment: reproduce chronological sequence of occupation
    #                     probabilities using Lanczos method for matrix exponential
    def test_alignment_ocs_lanczos(self):

        # matrix exponential method
        meth = "Lanczos"

        # list of quantum numbers F to use for reference check
        f_list = [0, 2, 4, 6, 8, 10, 12]

        # time step of propagation (ps)
        dt = 0.01

        # read basis
        fname_enr = 'etc/ocs_alignment/matelem/ocs_energies_j0_j30.rchm'
        psi = Psi(fname_enr, fmin=0, fmax=30, mmin=0, mmax=0, dm=2, df=2)

        # initalize wavepacket (f, m, id, ideg, coef)
        psi.j_m_id = (0, 0, 1, 1, 1.0 + 0.0j)

        # read tensor(s)
        fname_tens = 'etc/ocs_alignment/matelem/ocs_matelem_alpha_j<j1>_j<j2>.rchm'
        alpha = Etensor(fname_tens, psi)

        # read field (V/m)
        fname_field = 'etc/ocs_alignment/field.txt'
        f_field = open(fname_field,'r')
        field = [[0, 0, 100 * float(line.split()[3])] for line in f_field]
        f_field.close()


        # compute chronological sequence of occupation probabilities
        occu_probs = []
        for t_ind in range(int(self.t_end / dt)):
            print('t = ' + '{:03.2f}'.format(t_ind * dt) + ' ps')
            H = -0.5 * alpha * field[t_ind] # NOTE: only accounts for perturbation Hamiltonian
            psi = H.U(dt, psi, method=meth)
            if int(t_ind % 10) == 0:
                occu_probs.append([round(abs(psi.coefs[f][0])**2, 4) for f in f_list])


        # write chronological sequence of occupation probabilities
        fname_occu_probs = 'etc/ocs_alignment/' + meth + '_pop.txt'
        f_occu_probs = open(fname_occu_probs, 'w')
        f_occu_probs.write('T\M,F    0,0       0,2       0,4       0,6       0,8       0,10      0,12\n')
        for t_ind in range(len(occu_probs)):
            f_occu_probs.write('{:05.1f}'.format(t_ind * 10 * dt))
            for prob in occu_probs[t_ind]:
                f_occu_probs.write('    ' + '{:.4f}'.format(prob))
            f_occu_probs.write('\n')
        f_occu_probs.close()


        # read reference chronological sequence of occupation probabilities
        ref_occu_probs = []
        fname_ref_pop = 'etc/ocs_alignment/reference_pop.txt'
        f_ref_pop = open(fname_ref_pop, 'r')
        next(f_ref_pop)
        for line in f_ref_pop:
            split_line = line.split()
            probs = []
            for f_ind in range(len(f_list)):
                if 6 * (f_ind + 1) < len(split_line):
                    probs.append(float(split_line[2 + 6 * f_ind]))
                else:
                    probs.append(0)
            ref_occu_probs.append(probs)
        f_ref_pop.close()


        # check chronological sequence of occupation probabilites against reference
        for t_ind in range(len(occu_probs)):
            for f_ind in range(len(f_list)):
                occu_prob_devi = abs(occu_probs[t_ind][f_ind] - ref_occu_probs[t_ind][f_ind])
                if not occu_prob_devi <= self.occu_prob_tol:
                    print(str(t_ind * 10 * dt) + str(f_ind))
                self.assertTrue(occu_prob_devi <= self.occu_prob_tol)



if __name__ == "__main__":

    unittest.main()
