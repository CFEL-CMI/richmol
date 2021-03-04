# plugin for rigrot.Molecule
# adds centrifugal constants to rigrot.Molecule properties

class MoleculePlugin():

    def run(self, arg):
        arg.reduction = property(MoleculePlugin.get_reduction, MoleculePlugin.set_reduction)
        arg.dj = property(MoleculePlugin.get_dj, MoleculePlugin.set_dj)
        arg.djk = property(MoleculePlugin.get_djk, MoleculePlugin.set_djk)
        arg.dk = property(MoleculePlugin.get_dk, MoleculePlugin.set_dk)
        arg.d1 = property(MoleculePlugin.get_d1, MoleculePlugin.set_d1)
        arg.d2 = property(MoleculePlugin.get_d2, MoleculePlugin.set_d2)
        arg.hj = property(MoleculePlugin.get_hj, MoleculePlugin.set_hj)
        arg.hjk = property(MoleculePlugin.get_hjk, MoleculePlugin.set_hjk)
        arg.hkj = property(MoleculePlugin.get_hkj, MoleculePlugin.set_hkj)
        arg.hk = property(MoleculePlugin.get_hk, MoleculePlugin.set_hk)
        arg.h1 = property(MoleculePlugin.get_h1, MoleculePlugin.set_h1)
        arg.h2 = property(MoleculePlugin.get_h2, MoleculePlugin.set_h2)
        arg.h3 = property(MoleculePlugin.get_h3, MoleculePlugin.set_h3)

    def get_reduction(self):
        try:
            val = self.watson_reduction
        except AttributeError:
            raise AttributeError(f"Watson reduction is not specified") from None
        return val

    def set_reduction(self, val):
        print("call set_reduction")
        assert (val.lower() in ["a","s"]),f"Illegal value for Watson reduction = {val}, please use 'A' or 'S'"
        self.watson_reduction = val.lower()

    def get_dj(self):
        try:
            val = self.const_dj
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'dj' is not defined") from None
        return val
    def set_dj(self, val):
        self.const_dj = val

    def get_djk(self):
        try:
            val = self.const_djk
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'djk' is not defined") from None
        return val
    def set_djk(self, val):
        self.const_djk = val

    def get_dk(self):
        try:
            val = self.const_dk
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'dk' is not defined") from None
        return val
    def set_dk(self, val):
        self.const_dk = val

    def get_d1(self):
        try:
            val = self.const_d1
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'd1' is not defined") from None
        return val
    def set_d1(self, val):
        self.const_d1 = val

    def get_d2(self):
        try:
            val = self.const_d2
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'd2' is not defined") from None
        return val
    def set_d2(self, val):
        self.const_d2 = val

    def get_hj(self):
        try:
            val = self.const_hj
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'hj' is not defined") from None
        return val
    def set_hj(self, val):
        self.const_hj = val

    def get_hjk(self):
        try:
            val = self.const_hjk
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'hjk' is not defined") from None
        return val
    def set_hjk(self, val):
        self.const_hjk = val

    def get_hkj(self):
        try:
            val = self.const_hkj
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'hkj' is not defined") from None
        return val
    def set_hkj(self, val):
        self.const_hkj = val

    def get_hk(self):
        try:
            val = self.const_hk
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'hk' is not defined") from None
        return val
    def set_hk(self, val):
        self.const_hk = val

    def get_h1(self):
        try:
            val = self.const_h1
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'h1' is not defined") from None
        return val
    def set_h1(self, val):
        self.const_h1 = val

    def get_h2(self):
        try:
            val = self.const_h2
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'h2' is not defined") from None
        return val
    def set_h2(self, val):
        self.const_h2 = val

    def get_h3(self):
        try:
            val = self.const_h3
        except AttributeError:
            raise AttributeError(f"Centrifugal constant 'h3' is not defined") from None
        return val
    def set_h3(self, val):
        self.const_h3 = val
