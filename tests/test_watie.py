import unittest
import TestRunner
from richmol import watie
from hypothesis import given, example, settings, assume, strategies as st, HealthCheck
from hypothesis.extra.numpy import arrays
import numpy as np


class TestRigidMolecule(unittest.TestCase):

    # basic coordinates (s-enantiomer of camphor)
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


    # fail 'tensor'
    def fail_tens(self):
        mol = watie.RigidMolecule()
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
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
        mol.XYZ = self.XYZ
        tol = abs(np.linalg.norm(mol.imom())*np.finfo(float).eps)*10
        d,v = np.linalg.eigh(mol.imom())
        XYZ1 = np.dot(mol.XYZ["xyz"], v)[:,[2,0,1]]
        XYZ2 = np.dot(mol.XYZ["xyz"], v)
        mol.frame = "zxy,pas"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ1)<=tol))
        mol.frame = "pas,zxy,pas"
        self.assertTrue(np.all(np.abs(mol.XYZ["xyz"]-XYZ2)<=tol))


    # Test D2 symmetry: equal energies obtained with and without symmetry
    # use PAS frame and Hamiltonian built from rotational constants
    def test_energies_D2(self):
        mol = watie.RigidMolecule()
        mol.XYZ = self.XYZ
        mol.frame = "pas"
        J_list = [0] + np.random.randint(100, size=10)
        # without symmetry
        enr_all = []
        for J in J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            A, B, C = mol.ABC
            H = B * Jx2 + C * Jy2 + A * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat)
            enr_all += [e for e in enr]
        # using D2 symmetry
        enr_all_d2 = []
        for J in J_list:
            bas_d2 = watie.symmetrize(watie.SymtopBasis(J), sym="D2")
            for sym,bas in bas_d2.items():
                Jx2 = watie.Jxx(bas)
                Jy2 = watie.Jyy(bas)
                Jz2 = watie.Jzz(bas)
                A, B, C = mol.ABC
                H = B * Jx2 + C * Jy2 + A * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat)
                enr_all_d2 += [e for e in enr]
        tol = 1e-12
        self.assertTrue( all(abs(x-y)<tol for x,y in zip(sorted(enr_all),sorted(enr_all_d2))) )


    # Test C2v symmetry: equal energies obtained with and without symmetry
    # use PAS frame and Hamiltonian built from rotational constants
    def test_energies_C2v(self):
        mol = watie.RigidMolecule()
        mol.XYZ = self.XYZ
        mol.frame = "pas"
        J_list = [0] + np.random.randint(100, size=10)
        # without symmetry
        enr_all = []
        for J in J_list:
            bas = watie.SymtopBasis(J)
            Jx2 = watie.Jxx(bas)
            Jy2 = watie.Jyy(bas)
            Jz2 = watie.Jzz(bas)
            A, B, C = mol.ABC
            H = B * Jx2 + C * Jy2 + A * Jz2
            hmat = bas.overlap(H)
            enr, vec = np.linalg.eigh(hmat)
            enr_all += [e for e in enr]
        # using C2v symmetry
        enr_all_c2v = []
        for J in J_list:
            bas_d2 = watie.symmetrize(watie.SymtopBasis(J), sym="C2v")
            for sym,bas in bas_d2.items():
                Jx2 = watie.Jxx(bas)
                Jy2 = watie.Jyy(bas)
                Jz2 = watie.Jzz(bas)
                A, B, C = mol.ABC
                H = B * Jx2 + C * Jy2 + A * Jz2
                hmat = bas.overlap(H)
                enr, vec = np.linalg.eigh(hmat)
                enr_all_c2v += [e for e in enr]
        tol = 1e-12
        self.assertTrue( all(abs(x-y)<tol for x,y in zip(sorted(enr_all),sorted(enr_all_c2v))) )



if __name__=="__main__":
    #unittest.main()
    TestRunner.main()
    runner.run(my_test_suite)
