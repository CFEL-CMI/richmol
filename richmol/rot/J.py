import numpy as np
from basis import PsiTableMK


class J(PsiTableMK):
    """Basic class for rotational angular momentum operators"""

    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            try:
                x = arg.m
            except AttributeError:
                raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'm'") from None
            try:
                x = arg.k
            except AttributeError:
                raise AttributeError(f"'{arg.__class__.__name__}' has no attribute 'k'") from None
            PsiTableMK.__init__(self, arg.k, arg.m)


    def __mul__(self, arg):
        try:
            x = arg.m
            y = arg.k
            res = self.__class__(arg)
        except AttributeError:
            if np.isscalar(arg):
                res = J(self)
                res.k = res.k * arg
            else:
                raise TypeError(f"unsupported operand type(s) for '*': '{self.__class__.__name__}' " \
                        +f"and '{arg.__class__.__name__}'") from None
        return res


    __rmul__ = __mul__



class mol_Jp(J):
    """Molecular-frame J+ = Jx + iJy"""
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            table = self.k.table.copy()
            self.k.table['c'] = 0
            for ielem,(j,k) in enumerate(table['prim']):
                if abs(k-1) <= j:
                    fac = np.sqrt( j*(j+1) - k*(k-1) )
                    k2 = k - 1
                    jelem = np.where((table['prim']==(j,k2)).all(axis=1))[0][0]
                    self.k.table['c'][jelem,:] = table['c'][ielem,:] * fac
        except AttributeError:
            pass

Jp = mol_Jp


class mol_Jm(J):
    """Molecular-frame J- = Jx - iJy"""
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            table = self.k.table.copy()
            self.k.table['c'] = 0
            for ielem,(j,k) in enumerate(table['prim']):
                if abs(k+1) <= j:
                    fac = np.sqrt( j*(j+1) - k*(k+1) )
                    k2 = k + 1
                    jelem = np.where((table['prim']==(j,k2)).all(axis=1))[0][0]
                    self.k.table['c'][jelem,:] = table['c'][ielem,:] * fac
        except AttributeError:
            pass

Jm = mol_Jm


class mol_Jz(J):
    """Molecular-frame Jz"""
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.k.table['prim']):
                self.k.table['c'][ielem,:] *= k
        except AttributeError:
            pass

Jz = mol_Jz


class mol_JJ(J):
    """Molecular-frame J^2"""
    def __init__(self, arg=None):
        J.__init__(self, arg)
        try:
            for ielem,(j,k) in enumerate(self.k.table['prim']):
                self.k.table['c'][ielem,:] *= j*(j+1)
        except AttributeError:
            pass

JJ = mol_JJ


class mol_Jxx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.25 * ( mol_Jm(arg) * mol_Jm(arg) +  mol_Jm(arg) * mol_Jp(arg) \
                +  mol_Jp(arg) * mol_Jm(arg) +  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jxx = mol_Jxx


class mol_Jxy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( mol_Jm(arg) * mol_Jm(arg) -  mol_Jm(arg) * mol_Jp(arg) \
                +  mol_Jp(arg) * mol_Jm(arg) -  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jxy = mol_Jxy


class mol_Jyx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.25) * ( mol_Jm(arg) * mol_Jm(arg) +  mol_Jm(arg) * mol_Jp(arg) \
                -  mol_Jp(arg) * mol_Jm(arg) -  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jyx = mol_Jyx


class mol_Jxz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( mol_Jm(arg) * mol_Jz(arg) +  mol_Jp(arg) * mol_Jz(arg) )
            J.__init__(self, res)

Jxz = mol_Jxz


class mol_Jzx(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = 0.5 * ( mol_Jz(arg) * mol_Jm(arg) +  mol_Jz(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jzx = mol_Jzx


class mol_Jyy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = -0.25 * ( mol_Jm(arg) * mol_Jm(arg) -  mol_Jm(arg) * mol_Jp(arg) \
                -  mol_Jp(arg) * mol_Jm(arg) +  mol_Jp(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jyy = mol_Jyy


class mol_Jyz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( mol_Jm(arg) * mol_Jz(arg) -  mol_Jp(arg) * mol_Jz(arg) )
            J.__init__(self, res)

Jyz = mol_Jyz


class mol_Jzy(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = complex(0.0,0.5) * ( mol_Jz(arg) * mol_Jm(arg) -  mol_Jz(arg) * mol_Jp(arg) )
            J.__init__(self, res)

Jzy = mol_Jzy


class mol_Jzz(J):
    def __init__(self, arg=None):
        if arg is None:
            pass
        else:
            res = mol_Jz(arg) * mol_Jz(arg)
            J.__init__(self, res)

Jzz = mol_Jzz
