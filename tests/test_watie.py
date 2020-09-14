import unittest
import TestRunner
from richmol import watie
from hypothesis import given, example, settings, assume, strategies as st, HealthCheck
from hypothesis.extra.numpy import arrays
import numpy as np
import random
import warnings


# basic coordinates used in tests (s-enantiomer of camphor)
XYZ = ("angstrom", \
"O",     -2.547204,    0.187936,   -0.213755, \
"C",     -1.382858,   -0.147379,   -0.229486, \
"C",     -0.230760,    0.488337,    0.565230, \
"C",     -0.768352,   -1.287324,   -1.044279, \
"C",     -0.563049,    1.864528,    1.124041, \
"C",      0.716269,   -1.203805,   -0.624360, \
"C",      0.929548,    0.325749,   -0.438982, \
"C",      0.080929,   -0.594841,    1.638832, \
"C",      0.791379,   -1.728570,    0.829268, \
"C",      2.305990,    0.692768,    0.129924, \
"C",      0.730586,    1.139634,   -1.733020, \
"H",     -1.449798,    1.804649,    1.756791, \
"H",     -0.781306,    2.571791,    0.321167, \
"H",      0.263569,    2.255213,    1.719313, \
"H",      1.413749,   -1.684160,   -1.316904, \
"H",     -0.928638,   -1.106018,   -2.110152, \
"H",     -1.245108,   -2.239900,   -0.799431, \
"H",      1.816886,   -1.883799,    1.170885, \
"H",      0.276292,   -2.687598,    0.915376, \
"H",     -0.817893,   -0.939327,    2.156614, \
"H",      0.738119,   -0.159990,    2.396232, \
"H",      3.085409,    0.421803,   -0.586828, \
"H",      2.371705,    1.769892,    0.297106, \
"H",      2.531884,    0.195217,    1.071909, \
"H",      0.890539,    2.201894,   -1.536852, \
"H",      1.455250,    0.830868,   -2.487875, \
"H",     -0.267696,    1.035608,   -2.160680)


class TestRigidMolecule(unittest.TestCase):

    # fail 'tensor'
    def fail_tens(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        tens = np.random.rand(3,3)
        with self.assertRaises(ValueError):
            mol.tensor = ("b, la", tens)
        with self.assertRaises(ValueError):
            mol.tensor = ("  ,", tens)
        with self.assertRaises(ValueError):
            mol.tensor = ("  ", tens)
        with self.assertRaises(ValueError):
            mol.tensor = ("", tens)
        tens = np.random.rand(3,2)
        with self.assertRaises(ValueError):
            mol.tensor = ("tens", tens)
        tens = np.random.rand(3,4,3,1)
        with self.assertRaises(ValueError):
            mol.tensor = ("tens", tens)
        tens = np.random.rand(3,3,3)
        tens[:,:,:] = 0
        with self.assertRaises(ValueError):
            mol.tensor = ("tens", tens)
        tens[:,:,:] = np.finfo(float).max
        with self.assertRaises(ValueError):
            mol.tensor = ("tens", tens)
        tens[0,1,1] = np.NaN
        with self.assertRaises(ValueError):
            mol.tensor = ("tens", tens)


    # fail 'frame', without rotation defined by external tensor
    @settings(deadline=5000, max_examples=30)
    @given(s=st.text())
    @example(s="pas")
    @example(s="xyz")
    @example(s="zyx")
    @example(s="zzx")
    @example(s="zxy,pas")
    def fail_frame(self, s):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        sl = [v.strip().lower() for v in s.split(',')]
        if "pas" in sl:
            mol.frame = s
            # check if inertia tensor in the new frame is diagonal
            imom = mol.imom()
            tol = abs(np.linalg.norm(imom)*np.finfo(float).eps)*10
            self.assertTrue( np.all( np.abs(imom-np.diag(np.diag(imom))) < tol ) )
        elif any("".join(sorted(ss))=="xyz" for ss in sl):
            mol.frame = s
        else:
            with self.assertRaises((KeyError, AttributeError, AssertionError)):
                mol.frame = s


    # fail 'frame' with external tensor 'tensor', keep tensor symmetric (3,3) matrix
    @settings(deadline=5000, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    @given(st.text(), arrays(np.float64,(3,3)))
    def fail_frame_tens(self, x, y):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        yy = (y+y.T)*0.5 # make tensor symmetric
        try:
            mol.tensor = (x,yy)
        except ValueError:
            assume(False)
        try:
            mol.frame = x
        except (ValueError, RuntimeError):
            assume(False)
        # check if tensor in the new frame is diagonal
        xx = x.strip()
        tol = abs(np.linalg.norm(yy)*np.finfo(float).eps)*10
        self.assertTrue( np.all( np.abs(mol.tensor[xx]-np.diag(np.diag(mol.tensor[xx]))) < tol ) )


    # test PAS frame, for Cartesian coordinates
    def test_frame_pas(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        tol = abs(np.linalg.norm(mol.imom())*np.finfo(float).eps)*10
        d,v = np.linalg.eigh(mol.imom())
        XYZ1 = np.dot(mol.XYZ["xyz"], v)
        mol.frame = "pas"
        self.assertTrue(np.all(np.abs(mol.imom()-np.diag(d))<=tol))
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ1)<=tol))


    # test PAS frame, for tensors (3,3)
    @settings(deadline=5000, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    @given(st.text(), arrays(np.float64,(3,3)))
    def test2_frame_pas(self, x, y):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        tol = abs(np.linalg.norm(mol.imom()) * np.linalg.norm(y) * np.finfo(float).eps)
        try:
            mol.tensor = (x,y)
        except ValueError:
            assume(False)
        d,v = np.linalg.eigh(mol.imom())
        xx = x.strip()
        tens = np.dot(v.T,np.dot(mol.tensor[xx],v))
        mol.frame = "pas"
        self.assertTrue(np.all(np.abs(mol.tensor[xx]-tens)<=tol))


    # test PAS frame, for tensors (3,3,3)
    @settings(deadline=5000, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    @given(st.text(), arrays(np.float64,(3,3,3)))
    def test3_frame_pas(self, x, y):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        tol = abs(np.linalg.norm(mol.imom()) * np.linalg.norm(y) * np.finfo(float).eps)
        try:
            mol.tensor = (x,y)
        except ValueError:
            assume(False)
        d,v = np.linalg.eigh(mol.imom())
        xx = x.strip()
        tens = np.einsum('ia,jb,kc,abc->ijk',v.T,v.T,v.T,mol.tensor[xx])
        mol.frame = "pas"
        self.assertTrue(np.all(np.abs(mol.tensor[xx]-tens)<=tol))


    # test axes-permutation frame, for Cartesian coordinates
    def test_frame_perm(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        XYZ1 = mol.XYZ["xyz"][:,[2,0,1]]
        XYZ2 = XYZ1[:,[2,1,0]]
        XYZ3 = XYZ2[:,[0,2,1]]
        XYZ4 = XYZ3[:,[1,2,0]]
        XYZ5 = XYZ4[:,[1,2,0]][:,[0,1,2]][:,[1,0,2]]
        tol = abs(np.finfo(float).eps)*10
        mol.frame = "zxy"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ1)<=tol))
        mol.frame = "zyx"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ2)<=tol))
        mol.frame = "xzy"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ3)<=tol))
        mol.frame = "yzx"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ4)<=tol))
        mol.frame = "yxz,xyz,yzx"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ5)<=tol))


    # test axes-permutation + PAS frame, for Cartesian coordinates
    def test_frame_pas_perm(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        tol = abs(np.linalg.norm(mol.imom())*np.finfo(float).eps)*10
        d,v = np.linalg.eigh(mol.imom())
        XYZ1 = np.dot(mol.XYZ["xyz"], v)[:,[2,0,1]]
        XYZ2 = np.dot(mol.XYZ["xyz"], v)
        mol.frame = "zxy,pas"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ1)<=tol))
        mol.frame = "pas,zxy,pas"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ2)<=tol))


    # test 'linear'
    def test_linear(self):
        ntrials = 30
        for i in range(ntrials):
            # random vector
            e = np.random.rand(3)
            e = e/np.linalg.norm(3)
            # random number of atoms
            natoms = max([3,np.random.randint(30)])
            # random distances
            dist = np.random.rand(natoms)*20
            # random atomic labels
            labels = [random.choice("HOCNBPFSKI") for i in range(natoms)]
            # generate XYZ coordinates of atoms along the vector 'e'
            xyz = [val for r,lab in zip(dist,labels) for val in [lab]+list(r*e)]
            mol = watie.RigidMolecule()
            mol.XYZ = xyz
            self.assertTrue(mol.linear())



class TestPsiTable(unittest.TestCase):

    def fail_init(self):
        ntrials = 30
        for itrial in range(ntrials):
            nJ = np.random.randint(1,100)
            J_list = [0] + np.random.randint(1,100, size=nJ)
            prim = [(J,k) for J in J_list for k in range(-J,J+1)]
            nprim = len(prim)
            nstat = np.random.randint(nprim)
            stat_nelem = np.random.randint(10)
            stat = [np.random.randint(100, size=stat_nelem) for i in range(nstat) ]
            try:
                obj = watie.PsiTable(prim, stat)
            except (ValueError, AssertionError):
                continue
            with self.assertRaises(AttributeError):
                obj = watie.PsiTable(prim, stat, [1,2,3])
            with self.assertRaises(ValueError):
                coefs = np.random.rand(np.random.randint(1,nprim-1), np.random.randint(1,nstat))
                obj = watie.PsiTable(prim, stat, coefs)
            coefs = np.random.rand(nprim, nstat)
            obj = watie.PsiTable(prim, stat, coefs)


    def test_add_sub_mul(self):
        ntrials = 30
        tol = 1e-14
        for itrial in range(ntrials):
            # generate random prim, stat, and coefs
            nJ = np.random.randint(3,10)
            J_list = list(set([0] + np.random.randint(1,100, size=nJ)))
            prim = [(J,k) for J in J_list for k in range(-J,J+1)]
            nprim = len(prim)
            nstat = np.random.randint(1,nprim)
            stat_nelem = np.random.randint(2,10)
            stat = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat)))
            nstat = len(stat)
            coefs1 = np.random.rand(nprim, nstat)
            coefs2 = np.random.rand(nprim, nstat)
            # generate same tables with different values of coefs
            obj1 = watie.PsiTable(prim, stat, coefs1)
            obj2 = watie.PsiTable(prim, stat, coefs2)
            # test addition
            obj = obj1 + obj2
            self.assertTrue( (abs(obj.table['c'] - (obj1.table['c']+obj2.table['c'])) < tol).all() )
            # test subtraction
            obj = obj1 - obj2
            self.assertTrue( (abs(obj.table['c'] - (obj1.table['c']-obj2.table['c'])) < tol).all() )
            # test multiplication
            with self.assertRaises(TypeError):
                obj = obj1 * obj2
            const = random.random()
            obj = const * obj1
            self.assertTrue( (abs(obj.table['c'] - (obj1.table['c']*const)) < tol).all() )
            const = random.random()
            obj = obj1 * const
            self.assertTrue( (abs(obj.table['c'] - (obj1.table['c']*const)) < tol).all() )
            # fail addition and subtraction
            stat2 = [np.random.randint(100, size=stat_nelem) for i in range(nstat) ]
            stat2 = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat)))
            nstat2 = len(stat2)
            coefs2 = np.random.rand(nprim, nstat2)
            obj3 = watie.PsiTable(prim, stat2, coefs2)
            with self.assertRaises(ValueError):
                obj = obj1 + obj3
            with self.assertRaises(ValueError):
                obj = obj1 - obj3
            prim2 = [(J,k+1) for J in J_list for k in range(-J,J+1)]
            obj3 = watie.PsiTable(prim2, stat, coefs1)
            with self.assertRaises(ValueError):
                obj = obj1 + obj3
            with self.assertRaises(ValueError):
                obj = obj1 - obj3
            with self.assertRaises(AttributeError):
                obj = obj1 + 3
            with self.assertRaises(AttributeError):
                obj = 34.234 + obj1
            with self.assertRaises(AttributeError):
                obj = 100 - obj1
            with self.assertRaises(AttributeError):
                obj = obj1 - 34.5


    def test_append(self):
        ntrials = 30
        tol = 1e-12
        for itrial in range(ntrials):
            # generate random prim, stat, coef, and init PsiTable
            nJ = np.random.randint(3,10)
            J_list = list(set([0] + np.random.randint(1,100, size=nJ)))
            prim = [(J,k) for J in J_list for k in range(-J,J+1)]
            nprim = len(prim)
            nstat = np.random.randint(1,nprim)
            stat_nelem = np.random.randint(2,10)
            stat = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat)))
            nstat = len(stat)
            coefs = np.random.rand(nprim, nstat)
            psi = watie.PsiTable(prim, stat, coefs)
            # append object to itself and delete all duplicates
            psi2 = psi.append(psi, del_duplicate_stat=True)
            # check dimensions
            self.assertTrue( all( i == j for i,j in zip(psi2.table['c'].shape,[nprim,nstat]) ) )
            # check coefficient values
            prim2 = [tuple(p) for p in psi2.table['prim']]
            ind = [prim2.index(tuple(p)) for p in psi.table['prim']]
            self.assertTrue( (abs(psi2.table['c'][ind,:] - psi.table['c'][:,:]) < tol).all() )
            # append another object with different prim and stat
            #    but the same length of elements in prim and stat
            nJ = np.random.randint(3,10)
            J_list2 = [0] + np.random.randint(1,100, size=nJ)
            J_list2 = list(set([j for j in J_list2 if j not in J_list]))
            prim2 = [(J,k) for J in J_list2 for k in range(-J,J+1)]
            nprim2 = len(prim2)
            nstat2 = np.random.randint(1,nprim2)
            stat2 = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat2)))
            stat2 = [elem for elem in stat2 if elem not in stat]
            nstat2 = len(stat2)
            coefs2 = np.random.rand(nprim2, nstat2)
            psi2 = watie.PsiTable(prim2, stat2, coefs2)
            psi3 = psi.append(psi2, del_duplicate_stat=True)


    def test_overlap(self):
        ntrials = 30
        tol = 1e-12
        for itrial in range(ntrials):
            # generate random prim, stat, coef, and init PsiTable
            nJ = np.random.randint(3,10)
            J_list = list(set([0] + np.random.randint(1,100, size=nJ)))
            prim = [(J,k) for J in J_list for k in range(-J,J+1)]
            nprim = len(prim)
            nstat = np.random.randint(1,nprim)
            stat_nelem = np.random.randint(2,10)
            stat = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat)))
            nstat = len(stat)
            coefs = np.random.rand(nprim, nstat)
            psi = watie.PsiTable(prim, stat, coefs)
            # test overlap with itself
            ovlp = psi.overlap(psi)
            self.assertTrue( (abs(ovlp - np.dot(coefs.conj().T, coefs)) < tol).all() )
            # generate another psi2 not overlapping with psi
            nJ = np.random.randint(3,10)
            J_list2 = [0] + np.random.randint(1,100, size=nJ)
            J_list2 = list(set([j for j in J_list2 if j not in J_list]))
            prim2 = [(J,k) for J in J_list2 for k in range(-J,J+1)]
            nprim2 = len(prim2)
            nstat2 = np.random.randint(1,nprim2)
            stat2 = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat2)))
            stat2 = [elem for elem in stat2 if elem not in stat]
            nstat2 = len(stat2)
            coefs2 = np.random.rand(nprim2, nstat2)
            psi2 = watie.PsiTable(prim2, stat2, coefs2)
            # test zero overlap
            with self.assertWarns(UserWarning):
                ovlp = psi.overlap(psi2)
            self.assertTrue( (abs(ovlp)<tol).all() )


    def test_rotate(self):
        ntrials = 30
        tol = 1e-12
        for itrial in range(ntrials):
            # generate random prim, stat, coef, and init PsiTable
            nJ = np.random.randint(3,10)
            J_list = list(set([0] + np.random.randint(1,100, size=nJ)))
            prim = [(J,k) for J in J_list for k in range(-J,J+1)]
            nprim = len(prim)
            nstat = np.random.randint(1,nprim)
            stat_nelem = np.random.randint(2,10)
            stat = list(set(tuple(np.random.randint(100, size=stat_nelem)) for i in range(nstat)))
            nstat = len(stat)
            n = min([nstat,nprim])
            coefs = np.random.rand(n,n)
            coefs = 0.5 * (coefs + coefs.conj().T)   # Hermitian matrix of coefficients
            psi = watie.PsiTable(prim[:n], stat[:n], coefs)
            e,v = np.linalg.eigh(coefs)   # diagonalize coefs
            psi2 = psi.rotate(v.T)    # transform psi2 with eigenvectors of coefs
            coefs = np.dot(v.T, psi2.table['c'])
            self.assertTrue( ( abs(coefs - np.diag(np.diag(coefs))) < tol ).all() )



class TestEnergies(unittest.TestCase):

    J_list = [0] + np.random.randint(10, size=3)
    frame_ntrials = 30
    enr_tol = 1e-08

    # Test D2 symmetry: equal energies obtained with and without symmetry
    # use PAS frame and Hamiltonian built from rotational constants
    def test_energies_D2(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        mol.frame = "pas"
        Bx, By, Bz = mol.B
        # without symmetry
        enr_all = []
        for J in self.J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            H = Bx * Jx2 + By * Jy2 + Bz * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            enr_all += [e for e in enr]
        # using D2 symmetry
        enr_all_d2 = []
        for J in self.J_list:
            bas_d2 = watie.symmetrize(watie.SymtopBasis(J), sym="D2")
            for sym,bas in bas_d2.items():
                Jx2 = watie.Jxx(bas)
                Jy2 = watie.Jyy(bas)
                Jz2 = watie.Jzz(bas)
                H = Bx * Jx2 + By * Jy2 + Bz * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all_d2 += [e for e in enr]
        self.assertTrue( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr_all),sorted(enr_all_d2))) )


    # Test C2v symmetry: equal energies obtained with and without symmetry
    # use PAS frame and Hamiltonian built from rotational constants
    def test_energies_C2v(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        mol.frame = "pas"
        Bx, By, Bz = mol.B
        # without symmetry
        enr_all = []
        for J in self.J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            H = Bx * Jx2 + By * Jy2 + Bz * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            enr_all += [e for e in enr]
        # using C2v symmetry
        enr_all_c2v = []
        for J in self.J_list:
            bas_d2 = watie.symmetrize(watie.SymtopBasis(J), sym="C2v")
            for sym,bas in bas_d2.items():
                Jx2 = watie.Jxx(bas)
                Jy2 = watie.Jyy(bas)
                Jz2 = watie.Jzz(bas)
                H = Bx * Jx2 + By * Jy2 + Bz * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all_c2v += [e for e in enr]
        self.assertTrue( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr_all),sorted(enr_all_c2v))) )


    # Test Hamiltonian built from rotational constants: equal energies obtained for different choices
    # of PAS-frame axes, different quantization axes, and different symmetries.
    def test_energies_pas_abc_sym(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        # compute set of reference energies
        enr0 = []
        mol.frame = "pas"
        Bx, By, Bz = mol.B
        for J in self.J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            H = Bx * Jx2 + By * Jy2 + Bz * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            enr0 += [e for e in enr]
        # test different axes in PAS and different choices of quantization axes
        for itrial in range(self.frame_ntrials):
            # append new frame rotation
            frames = ["pas","xyz","zxy","zyx","xzy","yzx","yxz"]
            if np.mod(itrial,10)==0: # to keep the number of sequential frame rotations less than 10
                del mol.frame_rotation
                del mol.frame_type
                mol.frame = "pas"
            mol.frame = random.choice(frames)
            A, B, C = mol.B
            # compute energies for different J and different choices of quantization axes
            # without symmetry
            enr_all = [[],[],[],[],[],[]]
            for J in self.J_list:
                bas = watie.SymtopBasis(J)
                Jx2 = watie.Jxx(bas)
                Jy2 = watie.Jyy(bas)
                Jz2 = watie.Jzz(bas)
                #
                H = B * Jx2 + C * Jy2 + A * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[0] += [e for e in enr]
                #
                H = A * Jx2 + B * Jy2 + C * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[1] += [e for e in enr]
                #
                H = A * Jx2 + C * Jy2 + B * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[2] += [e for e in enr]
                #
                H = B * Jx2 + A * Jy2 + C * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[3] += [e for e in enr]
                #
                H = C * Jx2 + A * Jy2 + B * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[4] += [e for e in enr]
                #
                H = C * Jx2 + B * Jy2 + A * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all[5] += [e for e in enr]
            # check energies against reference
            self.assertTrue( all( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr))) \
                                  for enr in enr_all ) )
            # now repeat calculations using D2 symmetry
            enr_all = [[],[],[],[],[],[]]
            for J in self.J_list:
                bas_sym = watie.symmetrize(watie.SymtopBasis(J), sym="D2")
                for sym,bas in bas_sym.items():
                    Jx2 = watie.Jxx(bas)
                    Jy2 = watie.Jyy(bas)
                    Jz2 = watie.Jzz(bas)
                    #
                    H = B * Jx2 + C * Jy2 + A * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[0] += [e for e in enr]
                    #
                    H = A * Jx2 + B * Jy2 + C * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[1] += [e for e in enr]
                    #
                    H = A * Jx2 + C * Jy2 + B * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[2] += [e for e in enr]
                    #
                    H = B * Jx2 + A * Jy2 + C * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[3] += [e for e in enr]
                    #
                    H = C * Jx2 + A * Jy2 + B * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[4] += [e for e in enr]
                    #
                    H = C * Jx2 + B * Jy2 + A * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[5] += [e for e in enr]
            # check energies against reference
            self.assertTrue( all( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr))) \
                                  for enr in enr_all ) )
            # now repeat calculations using C2v symmetry
            enr_all = [[],[],[],[],[],[]]
            for J in self.J_list:
                bas_sym = watie.symmetrize(watie.SymtopBasis(J), sym="C2v")
                for sym,bas in bas_sym.items():
                    Jx2 = watie.Jxx(bas)
                    Jy2 = watie.Jyy(bas)
                    Jz2 = watie.Jzz(bas)
                    #
                    H = B * Jx2 + C * Jy2 + A * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[0] += [e for e in enr]
                    #
                    H = A * Jx2 + B * Jy2 + C * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[1] += [e for e in enr]
                    #
                    H = A * Jx2 + C * Jy2 + B * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[2] += [e for e in enr]
                    #
                    H = B * Jx2 + A * Jy2 + C * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[3] += [e for e in enr]
                    #
                    H = C * Jx2 + A * Jy2 + B * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[4] += [e for e in enr]
                    #
                    H = C * Jx2 + B * Jy2 + A * Jz2
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all[5] += [e for e in enr]
            # check energies against reference
            self.assertTrue( all( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr))) \
                                  for enr in enr_all ) )


    # Test Hamiltonian built from G-matrix: equal energies obtained for different choices of frames.
    # Don't use symmetry outside of PAS frame!
    def test_energies_frames_gmat(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        # compute set of reference energies
        enr0 = []
        mol.frame = "pas"
        Bx, By, Bz = mol.B
        for J in self.J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            H = Bx * Jx2 + By * Jy2 + Bz * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            enr0 += [e for e in enr]
        # now test different frames
        for itrial in range(self.frame_ntrials):
            # initialize new symmetric tensor (in case if frame="pol")
            tens = np.random.rand(3,3)
            tens = 0.5*(tens+tens.T)
            mol.tensor = ("pol", tens)
            # append new frame rotation
            frames = ["pas","xyz","zxy","zyx","xzy","yzx","yxz","pol"]
            if np.mod(itrial,10)==0: # to keep the number of sequential frame rotations less than 10
                del mol.frame_rotation
                del mol.frame_type
            mol.frame = random.choice(frames)
            gmat = mol.gmat()
            # compute energies for different J
            enr_all = []
            for J in self.J_list:
                bas = watie.SymtopBasis(J)
                H = 0.5 * ( gmat[0,0] * watie.Jxx(bas) + \
                            gmat[0,1] * watie.Jxy(bas) + \
                            gmat[0,2] * watie.Jxz(bas) + \
                            gmat[1,0] * watie.Jyx(bas) + \
                            gmat[1,1] * watie.Jyy(bas) + \
                            gmat[1,2] * watie.Jyz(bas) + \
                            gmat[2,0] * watie.Jzx(bas) + \
                            gmat[2,1] * watie.Jzy(bas) + \
                            gmat[2,2] * watie.Jzz(bas) )
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat.real)
                enr_all += [e for e in enr]
            # check energies against reference
            self.assertTrue( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr_all))) )
            del mol.tens["pol"] # because we can't initialize twice tensor with the same name


    # Test Hamiltonian built from G-matrix: equal energies obtained for different symmetries,
    # note that it is valid only if PAS frame is used.
    def test_energies_pas_gmat_sym(self):
        mol = watie.RigidMolecule()
        mol.XYZ = XYZ
        # compute set of reference energies
        enr0 = []
        mol.frame = "pas"
        Bx, By, Bz = mol.B
        for J in self.J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            H = Bx * Jx2 + By * Jy2 + Bz * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat.real)
            enr0 += [e for e in enr]
        # now test different axes permutations in PAS frame
        for itrial in range(self.frame_ntrials):
            frames = ["pas","xyz","zxy","zyx","xzy","yzx","yxz"]
            if np.mod(itrial,10)==0: # to keep the number of sequential frame rotations less than 10
                del mol.frame_rotation
                del mol.frame_type
                mol.frame = "pas"
            mol.frame = random.choice(frames)
            gmat = mol.gmat()
            # compute energies for different J, use D2 symmetry
            enr_all = []
            for J in self.J_list:
                bas_sym = watie.symmetrize(watie.SymtopBasis(J), sym="D2")
                for sym,bas in bas_sym.items():
                    H = 0.5 * ( gmat[0,0] * watie.Jxx(bas) + \
                                gmat[0,1] * watie.Jxy(bas) + \
                                gmat[0,2] * watie.Jxz(bas) + \
                                gmat[1,0] * watie.Jyx(bas) + \
                                gmat[1,1] * watie.Jyy(bas) + \
                                gmat[1,2] * watie.Jyz(bas) + \
                                gmat[2,0] * watie.Jzx(bas) + \
                                gmat[2,1] * watie.Jzy(bas) + \
                                gmat[2,2] * watie.Jzz(bas) )
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all += [e for e in enr]
            # check energies against reference
            self.assertTrue( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr_all))) )
            # compute energies for different J, use C2v symmetry
            enr_all = []
            for J in self.J_list:
                bas_sym = watie.symmetrize(watie.SymtopBasis(J), sym="C2v")
                for sym,bas in bas_sym.items():
                    H = 0.5 * ( gmat[0,0] * watie.Jxx(bas) + \
                                gmat[0,1] * watie.Jxy(bas) + \
                                gmat[0,2] * watie.Jxz(bas) + \
                                gmat[1,0] * watie.Jyx(bas) + \
                                gmat[1,1] * watie.Jyy(bas) + \
                                gmat[1,2] * watie.Jyz(bas) + \
                                gmat[2,0] * watie.Jzx(bas) + \
                                gmat[2,1] * watie.Jzy(bas) + \
                                gmat[2,2] * watie.Jzz(bas) )
                    hmat = bas.overlap(H)
                    enr, vec = np.linalg.eigh(hmat.real)
                    enr_all += [e for e in enr]
            # check energies against reference
            self.assertTrue( all(abs(x-y)<self.enr_tol for x,y in zip(sorted(enr0),sorted(enr_all))) )





class TestCartTensor(unittest.TestCase):

    @settings(deadline=5000, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    @given(arrays(np.float64,3))
    def fail_init1(self, x):
        try:
            tens = watie.CartTensor(x)
        except ValueError:
            assume(False)

    @settings(deadline=5000, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
    @given(arrays(np.float64,(3,3)))
    def fail_init2(self, x):
        try:
            tens = watie.CartTensor(x)
        except ValueError:
            assume(False)


if __name__=="__main__":
    #unittest.main()
    TestRunner.main()
    runner.run(my_test_suite)
