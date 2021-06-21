from mendeleev import element
import numpy as np
from scipy import constants
import re


_reading_funcs = dict()

def register(arg):
    if callable(arg):
        raise ValueError(f"please specify type of argument in @register('<type>') for function '{arg.__name__}'") from None
    else:
        if isinstance(arg, (list, tuple)):
            typ = [t for t in arg]
        else:
            typ = [arg]
    def _decorator(func):
        for t in typ:
            _reading_funcs[t] = func
        return func
    return _decorator


def read(arg):
    if type(arg).__name__ in _reading_funcs:
        return _reading_funcs[type(arg).__name__](arg)
    else:
        raise ValueError(f"read function for argument type '{type(arg)}' is not available") from None


def atomic_data(atom_label):
    """Given atom label, returns atomic properties.
    To specify different isotopologues: 'H2' (deuterium), 'C13', 'N15', etc.
    """
    r = re.compile("([a-zA-Z]+)([0-9]+)")
    m = r.match(atom_label)
    if m is None:
        atom = atom_label
        mass_number = 0
    else:
        atom = m.group(1)
        mass_number = int(m.group(2))
    elem = element(atom)
    if mass_number==0:
        mass_number = int(round(elem.mass,0))
    try:
        ind = [iso.mass_number for iso in elem.isotopes].index(mass_number)
    except ValueError:
        raise ValueError(f"can't find '{atom}' isotope '{mass_number}' in mendeleev database") from None
    mass = [iso.mass for iso in elem.isotopes][ind]
    return {"mass":mass}


@register('str')
def read_xyz_file(file_name):
    """Reads atomic coordinates from XYZ file"""
    xyz = []
    mass = []
    label = []
    with open(file_name, 'r') as fl:
        line = fl.readline()
        natoms = float(line.split()[0])
        comment = fl.readline()
        for line in fl:
            w = line.split()
            atom_label = w[0]
            try:
                x,y,z = (float(ww) for ww in w[1:])
            except ValueError:
                raise ValueError(f"failed to read XYZ file {file_name}") from None
            atom_mass = atomic_data(atom_label.upper())["mass"]
            xyz.append([x,y,z])
            mass.append(atom_mass)
            label.append(atom_label)
    atoms = np.array( [(lab, mass, cart) for lab,mass,cart in zip(label,mass,xyz)], \
                      dtype=[('label','U10'),('mass','f8'),('xyz','f8',(3))] )
    return atoms


@register(('list','tuple'))
def read_list(arg):
    """Reads atomic coordinates from input list"""
    xyz = []
    mass = []
    label = []
    to_angstrom = 1
    for ielem,elem in enumerate(arg):
        if isinstance(elem, str):
            if elem[:4].lower() == "bohr":
                to_angstrom = constants.value('Bohr radius') * 1e10
            elif elem[:4].lower() == "angs":
                to_angstrom = 1
            else:
                atom_label = elem
                atom_mass = atomic_data(atom_label.title())["mass"]
                try:
                    x,y,z = (float(val) for val in arg[ielem+1:ielem+4])
                except ValueError:
                    raise ValueError(f"failed to read coordinates for atom '{atom_label}'") from None
                xyz.append([float(val)*to_angstrom for val in (x,y,z)])
                mass.append(atom_mass)
                label.append(atom_label)
    atoms = np.array( [(lab, mass, cart) for lab,mass,cart in zip(label,mass,xyz)], \
                      dtype=[('label','U10'),('mass','f8'),('xyz','f8',(3))] )
    return atoms

