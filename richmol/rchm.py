import h5py
import numpy as np
import re
import os as opersys
from scipy.sparse import csr_matrix
import time
import datetime
from richmol.rot import molecule
from richmol.rot import labtens
from richmol.rot import mol_tens
from richmol.rot import solution
from richmol import field
import inspect
import json
from collections.abc import Mapping


def J_group_key(J1, J2):
    return 'J:' + str(round(float(J1), 1)) + ',' + str(round(float(J2), 1))


def sym_group_key(sym1, sym2):
    return 'sym:' + str(sym1) + ',' + str(sym2)


def store(filename, obj, *args, **kwargs):
    """Stores objects in richmol hdf5 file

    Args:
        filename : str
            Name of hdf5 file
        obj : object
            Object to store, one of the following:
                rot.molecule.Molecule
                rot.labtens.LabTensor
                field.CarTens
    """
    if isinstance(obj, molecule.Molecule):
        add_rig_molecule(filename, obj, *args, **kwargs)
    elif isinstance(obj, (labtens.LabTensor, field.CarTens)):
        add_tensor(filename, obj, *args, **kwargs)
    else:
        raise TypeError(f"unsupported type: '{type(obj)}'") from None


def add_rig_molecule(filename, mol, comment=None, replace=False):
    """Stores molecular data

    Args:
        filename : str
            Name of hdf5 file
        mol : rot.molecule.Molecule
            Molecular data
        comment : str
            Description of molecular data
        replace : bool
            If True, the existing dataset will be replaced
    """
    if not isinstance(mol, molecule.Molecule):
        raise TypeError(f"bad argument type for molecule") from None

    with h5py.File(filename, 'a') as fl:

        # create group
        if 'molecule' in fl:
            if replace is True:
                del fl['molecule']
            else:
                raise RuntimeError(f"found existing molecule dataset in '{filename}', " + \
                                   f"use replace=True to replace it") from None
        group = fl.create_group('molecule')

        # date/time
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        doc = date.replace('\n','')

        # user comment
        if comment is not None:
            doc = doc + " " + " ".join(elem for elem in comment.split())

        group.attrs['__doc__'] = doc

        # store XYZ
        try:
            xyz = mol.XYZ
            for name in xyz.dtype.names:
                try:
                    group.attrs[name] = xyz[name]
                except TypeError: # atom labels are U strings
                    group.attrs[name] = [str.encode(elem) for elem in xyz[name]]
        except AttributeError:
            pass

        # store rotational constants
        try:
            if mol.linear:
                try:
                    group.attrs['ABC'] = mol.ABC
                except Exception:
                    group.attrs['Gmat'] = mol.gmat()
            else:
                try:
                    group.attrs['B'] = mol.B
                except Exception:
                    pass
        except AttributeError:
            pass

        # store molecular tensors (dipole moment, polarizability, etc.)
        for tens in mol_tens._tensors.keys():
            try:
                group.attrs[tens] = getattr(mol, tens)
            except AttributeError:
                pass

        # store custom Hamiltonians and their parameters
        try:
            watson = mol.watson
            group.attrs['watson'] = str.encode(watson)
            for const in solution._constants[watson]:
                try:
                    group.attrs[tens] = getattr(mol, const)
                except AttributeError:
                    pass
        except AttributeError:
            pass


def read_molecule(filename):
    """Reads molecular data

    Args:
        filename : str
            Name of hdf5 file

    Returns:
        mol : object
            Contains molecular parameters, such as Cartesian cooridnates of atoms,
            molecular property tensors, rotational constants, Hamiltonian parameters, etc.
    """
    with h5py.File(filename, 'a') as fl:
        try:
            group = fl['molecule']
        except KeyError:
            raise KeyError(f"file '{filename}' has no molecule dataset") from None
        mol = type("molecule", (object,), {key:val for key,val in group.attrs.items()})
        return mol


def add_tensor(filename, tens, name=None, comment=None, replace=False, thresh=None):
    """Stores Cartesian tensor operator

    Args:
        filename : str
            Name of hdf5 file
        tens : labtens.LabTensor or field.CarTens
            Cartesian tensor object
        name : str
            Name of data group, by default, the name of argument 'tens' is used
        comment : str
            Description of tensor
        replace : bool
            If True, the existing dataset will be replaced
        thresh : float
            Threshold for neglecting matrix elements when writing into file
    """
    if not isinstance(tens, (labtens.LabTensor, field.CarTens)):
        raise TypeError(f"bad argument type for tensor") from None

    # group name
    if name is None:
        name_ = retrieve_name(tens)
    else:
        name_ = name

    with h5py.File(filename, 'a') as fl:

        # create group
        if name_ in fl:
            if replace is True:
                del fl[name_]
            else:
                raise RuntimeError(f"found existing tensor dataset '{name_}' in '{filename}', " + \
                                   f"use replace=True to replace it") from None
        group = fl.create_group(name_)

        # date/time
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        doc = date.replace('\n','')

        # user comment
        if comment is not None:
            doc = doc + " " + " ".join(elem for elem in comment.split())

        group.attrs['__doc__'] = doc

        # store tensor attributes except M and K tensors, and few others
        attrs = list(set(vars(tens).keys()) - set(['mmat', 'kmat', 'molecule']))
        for attr in attrs:
            val = getattr(tens, attr)
            try:
                group.attrs[attr] = val
            except TypeError:
                # store dictionary as json
                jd = json.dumps(val)
                group.attrs[attr+'_JSON'] = jd

        # store matrix elements, M and K tensors

        # loop over pairs of coupled J quanta
        for (J1, J2) in list(set(tens.mmat.keys()) & set(tens.kmat.keys())):

            mmat_sym = tens.mmat[(J1, J2)]
            kmat_sym = tens.kmat[(J1, J2)]

            # loop over pairs of coupled symmetries
            for (sym1, sym2) in list(set(mmat_sym.keys()) & set(kmat_sym.keys())):

                # store K-matrix

                # remove elements smaller that 'thresh'
                kmat = kmat_sym[(sym1, sym2)]
                kmat_ = [k for k in kmat.values()]
                if thresh is not None:
                    for k in kmat_:
                        mask = np.abs(k.data) < thresh
                        k.data[mask] = 0
                        k.eliminate_zeros()

                data = [k.data for k in kmat_ if k.nnz > 0]
                indices = [k.indices for k in kmat_ if k.nnz > 0]
                indptr = [k.indptr for k in kmat_ if k.nnz > 0]
                shape = [k.shape for k in kmat_ if k.nnz > 0]
                irreps = [key for key,k in zip(kmat.keys(), kmat_) if k.nnz > 0]
                if len(data) > 0:
                    try:
                        group_j = group[J_group_key(J1, J2)]
                    except:
                        group_j = group.create_group(J_group_key(J1, J2))
                    try:
                        group_sym = group_j[sym_group_key(sym1, sym2)]
                    except:
                        group_sym = group_j.create_group(sym_group_key(sym1, sym2))
                    group_sym.create_dataset('kmat_data', data=np.concatenate(data))
                    group_sym.create_dataset('kmat_indices', data=np.concatenate(indices))
                    group_sym.create_dataset('kmat_indptr', data=np.concatenate(indptr))
                    group_sym.attrs['kmat_nnz'] = [len(dat) for dat in data]
                    group_sym.attrs['kmat_nind'] = [len(ind) for ind in indices]
                    group_sym.attrs['kmat_nptr'] = [len(ind) for ind in indptr]
                    group_sym.attrs['kmat_irreps'] = irreps
                    group_sym.attrs['kmat_shape'] = shape

                # store M-matrix

                # remove elements smaller that 'thresh'
                mmat = mmat_sym[(sym1, sym2)]
                mmat_ = [m for mcart in mmat.values() for m in mcart.values()]
                if thresh is not None:
                    for m in mmat_:
                        mask = np.abs(m.data) < thresh
                        m.data[mask] = 0
                        m.eliminate_zeros()

                data = [m.data for m in mmat_ if m.nnz > 0]
                indices = [m.indices for m in mmat_ if m.nnz > 0]
                indptr = [m.indptr for m in mmat_ if m.nnz > 0]
                shape = [m.shape for m in mmat_ if m.nnz > 0]
                irreps_cart = [(key1, key2) for key2 in mmat.keys() for key1 in mmat[key2].keys()]
                irreps_cart = [(irrep, cart) for (irrep,cart),m in zip(irreps_cart,mmat_) if m.nnz > 0]
                if len(data) > 0:
                    try:
                        group_j = group[J_group_key(J1, J2)]
                    except:
                        group_j = group.create_group(J_group_key(J1, J2))
                    try:
                        group_sym = group_j[sym_group_key(sym1, sym2)]
                    except:
                        group_sym = group_j.create_group(sym_group_key(sym1, sym2))
                    group_sym.create_dataset('mmat_data', data=np.concatenate(data))
                    group_sym.create_dataset('mmat_indices', data=np.concatenate(indices))
                    group_sym.create_dataset('mmat_indptr', data=np.concatenate(indptr))
                    group_sym.attrs['mmat_nnz'] = [len(dat) for dat in data]
                    group_sym.attrs['mmat_nind'] = [len(ind) for ind in indices]
                    group_sym.attrs['mmat_nptr'] = [len(ind) for ind in indptr]
                    group_sym.attrs['mmat_irreps_cart'] = irreps_cart
                    group_sym.attrs['mmat_shape'] = shape


def json_to_dict(node):
    """USE WITH CARE: can delete duplicate keys, for example, "1" and 1

    Converts keys in nested dictionary to integers and floats where possible.
    Used to convert JSON dictionaries read from HDF5 file back to original form
    """
    out = dict()
    for key, item in node.items():
        if isinstance(item, Mapping):
            item_ = json_to_dict(item)
        else:
            item_ = item
        try:
            if key.find('.') == -1:
                key_ = int(key)
            else:
                key_ = float(key)
        except ValueError:
            key_ = key
        out[key_] = item_
    return out


def inspect_file(filename):
    """Returns information about all data groups stored in file

    Arg:
        filename : str
            Name of hdf5 file

    Returns:
        datagroup : dict
            Dictionary of data groups stored in file.
    """
    J_key_re = re.sub(r'1.0', '\d+\.\d+', J_group_key(1, 1))
    sym_key_re = re.sub(r'A', '\w+', sym_group_key('A', 'A'))

    datagroup = dict()

    with h5py.File(filename, 'a') as fl:
        for name in fl.keys():
            # collect attributes
            attrs = dict()
            for key, val in fl[name].attrs.items():
                if key.find('_JSON') == -1:
                    attrs[key] = val
                else:
                    jl = json.loads(val) # problem with JSON is that it converts dict keys into strings
                    if isinstance(jl, Mapping):
                        # solution: convert dict string keys into integers and floats (where possible)
                        jl = json_to_dict(jl)
                    attrs[key.replace('_JSON', '')] = jl
            # add object to datagroup dictionary
            datagroup[name] = type(name, (object,), attrs)

    return datagroup


def read_tensor(filename, name):
    """Reads Cartesian tensor operator

    Args:
        filename : str
            Name of hdf5 file
        name : str
            Name of tensor data group

    Returns:
        tens : object
            Cartesian tensor object
    """
    J_key_re = re.sub(r'1.0', '\d+\.\d+', J_group_key(1, 1))
    sym_key_re = re.sub(r'A', '\w+', sym_group_key('A', 'A'))

    # read data group attributes
    data_groups = inspect_file(filename)

    # init Cartesian tensor
    tens = field.CarTens()
    try:
        tens_data = data_groups[name]
        tens.__dict__.update(tens_data.__dict__)
    except KeyError:
        raise KeyError(f"file '{filename}' has no tensor dataset '{name}'")

    # load M and K tensors

    with h5py.File(filename, 'a') as fl:
        try:
            group = fl[name]
        except KeyError:
            raise KeyError(f"file '{filename}' has no tensor dataset '{name}'") from None

        tens.kmat = dict()
        tens.mmat = dict()

        # search for J groups
        for key in group.keys():

            # pair of coupled J quanta
            if re.match(J_key_re, key):
                Jpair = re.findall(f'\d+.\d+', key)
                J1, J2 = (round(float(elem), 1) for elem in Jpair)
                group_j = group[key]

                # search for symmetry groups
                for key2 in group_j.keys():

                    # pair of coupled symmetries
                    if re.match(sym_key_re, key2):
                        sympair = re.findall(f'\w+', key2)
                        _, sym1, sym2 = sympair
                        group_sym = group_j[key2]

                        # read K-matrix

                        kmat = None
                        try:
                            nnz = group_sym.attrs['kmat_nnz']
                            nind = group_sym.attrs['kmat_nind']
                            nptr = group_sym.attrs['kmat_nptr']
                            shape = group_sym.attrs['kmat_shape']
                            irreps = group_sym.attrs['kmat_irreps']
                            data = np.split(group_sym['kmat_data'], np.cumsum(nnz))[:-1]
                            indices = np.split(group_sym['kmat_indices'], np.cumsum(nind))[:-1]
                            indptr = np.split(group_sym['kmat_indptr'], np.cumsum(nptr))[:-1]
                            kmat = {irrep : csr_matrix((dat, ind, ptr), shape=sh)
                                    for irrep,dat,ind,ptr,sh in zip(irreps,data,indices,indptr,shape)}
                        except KeyError:
                            pass

                        # add K-matrix to tensor object
                        if kmat is not None:
                            try:
                                tens.kmat[(J1, J2)][(sym1, sym2)] = kmat
                            except KeyError:
                                tens.kmat[(J1, J2)] = {(sym1, sym2) : kmat}

                        # read M-matrix

                        mmat = None
                        try:
                            nnz = group_sym.attrs['mmat_nnz']
                            nind = group_sym.attrs['mmat_nind']
                            nptr = group_sym.attrs['mmat_nptr']
                            shape = group_sym.attrs['mmat_shape']
                            irreps_cart = group_sym.attrs['mmat_irreps_cart']
                            data = np.split(group_sym['mmat_data'], np.cumsum(nnz))[:-1]
                            indices = np.split(group_sym['mmat_indices'], np.cumsum(nind))[:-1]
                            indptr = np.split(group_sym['mmat_indptr'], np.cumsum(nptr))[:-1]
                            mmat = {irr_car[1] : {irr_car[0] : csr_matrix((dat, ind, ptr), shape=sh) }
                                    for irr_car,dat,ind,ptr,sh in zip(irreps_cart,data,indices,indptr,shape)}
                        except KeyError:
                            pass

                        # add M-matrix to tensor object
                        if mmat is not None:
                            try:
                                tens.mmat[(J1, J2)][(sym1, sym2)] = mmat
                            except KeyError:
                                tens.mmat[(J1, J2)] = {(sym1, sym2) : mmat}

    tens.check_attrs()
    return tens


def read_states(filename, **kwargs):
    """Reads basis states information from old-format richmol states file

    Args:
        filename : str
            Name of richmol states file

    Kwargs:
        jmin, jmax : int or float
            Min and max values of quantum number J of total angular momentum
        jlist : list
            List of J values, if present, overrides jmin and jmax
        mmin, mmax : int or float
            Min and max values of quantum number m of Z projection of total
            angular momentum
        mlist : list
            List of m values, if present, overrides mmin and mmax
        mdict : dict
            Dictionary mdict[J] -> list, contains list of m quantum numbers
            for different values of J, if present, overrides mlist
        emin, emax : float
            Min and max values of state energy
        symlist : list
            List of state symmetries
        symdic : dict:
            Dictionary symdic[J] -> list, contains list of symmetries
            for different values of J, if present, overrides symlist

    Returns:
        tens : field.CarTens
            Cartesian tensor object containing diagonal representation
            of field-free Hamiltonain
    """

    # read states file

    energy = dict()
    assign = dict()
    map_kstates = dict()

    with open(filename, 'r') as fl:
        for line in fl:
            w = line.split()
            J = round(float(w[0]),1)
            id = np.int64(w[1])
            sym = w[2]
            ndeg = int(w[3])
            enr = float(w[4])
            qstr = ' '.join([w[i] for i in range(5,len(w))])

            # apply state filters
            if 'jlist' in kwargs:
                if J not in [round(float(elem),1) for elem in kwargs['jlist']]:
                    continue
            else:
                if 'jmin' in kwargs:
                    if J < kwargs['jmin']:
                        continue
                if 'jmax' in kwargs:
                    if J > kwargs['jmax']:
                        continue
            if 'emin' in kwargs:
                if enr < kwargs['emin']:
                    continue
            if 'emax' in kwargs:
                if enr > kwargs['emax']:
                    continue
            if 'symdic' in kwargs:
                try:
                    if sym.lower() not in [elem.lower() for elem in kwargs['symdic'][J]]:
                        continue
                except KeyError:
                    pass
            elif 'symlist' in kwargs:
                if sym not in kwargs['symlist']:
                    continue

            if J not in energy:
                energy[J] = dict()
                assign[J] = dict()

            for ideg in range(ndeg):
                try:
                    energy[J][sym].append(enr)
                    assign[J][sym].append(qstr)
                except KeyError:
                    energy[J][sym] = [enr]
                    assign[J][sym] = [qstr]

                # mapping between J,id,ideg and basis set index running
                # within the group of states sharing the same J and symmetry
                map_kstates[(J, id, ideg)] = [len(energy[J][sym])-1, sym]

    # check how many states have passed the filters

    if len(list(energy.keys())) == 0:
        raise Exception(f"zero number of states, perhaps selection filters cast out states") from None

    if 'jlist' in kwargs:
        j_none = [J for J in kwargs['jlist'] if J not in energy]
        if len(j_none) > 0:
            raise Exception(f"states with the following J quanta were not found: {j_none}") from None

    if 'symdic' in kwargs:
        sym_none = {J : [sym for sym in kwargs['symdic'][J] if sym not in energy[J].keys()]
                    for J in kwargs['symdic'].keys()}
    elif 'symlist' in kwargs:
        symlist = set([sym for J in energy.keys() for sym in energy[J].keys()])
        sym_none = [sym for sym in kwargs['symlist'] if sym not in symlist]
    else:
        sym_none = []
    if len(sym_none) > 0:
        raise Exception(f"states with the following symmetries were not found: {sym_none}") from None

    # generate lists of m quanta for different J

    if 'mdict' in kwargs:
        mdict = {J : [round(float(elem),1) for elem in kwargs['mdict'][J]]
                 for J in list(energy.keys() & kwargs['mdict'].keys)}
        # for J values that are not present in kwargs['mdict'] set m ranges -J..J
        j_out = [J for J in energy.keys() if J not in mdict]
        mdict += {J : [round(float(m),1) for m in np.linalg(-J, J, int(2*J)+1)] for J in j_out}
    elif 'mlist' in kwargs:
        mdict = {J : [round(float(m),1) for m in kwargs['mlist'] if abs(round(float(m),1))<=J]
                 for J in energy.keys()}
    else:
        mdict = dict()
        mmin = None
        mmax = None
        if 'mmin' in kwargs:
            mmin = round(float(kwargs['mmin']),1)
        if 'mmax' in kwargs:
            mmax = round(float(kwargs['mmax']),1)
        if mmin is not None and mmax is not None:
            assert (mmin<=mmax), f"'mmin' = {mmin} > 'mmax' = {mmax}"
        for J in energy.keys():
            if mmin is None:
                m1 = -J
            else:
                m1 = max(-J, mmin)
            if mmax is None:
                m2 = J
            else:
                m2 = min(J, mmax)
            if m1>m2: continue
            mdict[J] = [round(float(m),1) for m in np.linspace(m1, m2, m2-m1+1)]

    # delete entries with zero length
    mdict = {key : val for key,val in mdict.items() if len(val)>0}
    # check if m-quanta filters cast out some of J quanta
    j_none = [J for J in energy.keys() if J not in mdict]
    if len(j_none) > 0:
        raise Exception(f"m-quanta filters cast out following J quanta: {j_none}") from None

    # generate mapping beteween m quanta and basis set index
    map_mstates = {(J, m) : ind_m for J in energy.keys() for ind_m,m in enumerate(mdict[J]) }

    # generate field.CarTens object attributes

    Jlist = list(energy.keys())
    symlist = {J : [sym for sym in energy[J].keys()] for J in Jlist}
    dim_m = {J : {sym : len(mdict[J]) for sym in symlist[J]} for J in Jlist}
    dim_k = {J : {sym : len(energy[J][sym]) for sym in symlist[J]} for J in Jlist}
    dim = {J : {sym : dim_m[J][sym] * dim_k[J][sym] for sym in symlist[J]} for J in Jlist}
    assign_m = {J : {sym : ["%4.1f"%m for m in mdict[J]] 
                for sym in symlist[J]} for J in Jlist}
    assign_k = {J : {sym : assign[J][sym] for sym in symlist[J]} for J in Jlist}

    cart = '0'
    os = [(0,0)]
    rank = 0

    Jlist1 = Jlist
    Jlist2 = Jlist
    symlist1 = symlist
    symlist2 = symlist
    dim1 = dim
    dim2 = dim
    dim_m1 = dim_m
    dim_m2 = dim_m
    dim_k1 = dim_k
    dim_k2 = dim_k
    assign_k1 = assign_k
    assign_k2 = assign_k
    assign_m1 = assign_m
    assign_m2 = assign_m

    mmat = {(J, J) : {(sym, sym) : {'0' : {0 : csr_matrix(np.eye(len(mdict[J])), dtype=np.complex128)}}
            for sym in symlist[J]} for J in Jlist}

    kmat = {(J, J) : {(sym, sym) : {'0' : {0 : csr_matrix(np.diag(energy[J][sym]), dtype=np.complex128)}}
            for sym in symlist[J]} for J in Jlist}

    # initialize field.CarTens object

    loc = locals()
    tens = field.CarTens()
    tens.attr_list += ('map_kstates', 'map_mstates')
    for name in tens.attr_list:
        setattr(tens, name, loc[name])
    tens.check_attrs()

    return tens


def read_trans(states, filename, thresh=None):
    """Reads matrix elements of Cartesian tensor from old-format Richmol
    matrix elements files

    Args:
        states : field.CarTens
            Field-free basis states (see read_states)
        filename : str
            In old format, matrix elements for different bra and ket J quantum
            numbers are stored in separate files. Argument filename provides
            a template for generating the names of these files.
            For example, for filename = "matelem_alpha_j<j1>_j<j2>.rchm",
            the following files will be searched: matelem_alpha_j0_j0.rchm,
            matelem_alpha_j0_j1.rchm, matelem_alpha_j0_j2.rchm,
            matelem_alpha_j1_j1.rchm, etc., where <j1> and <j2> are replaced
            by integer numbers running through all J quanta spanned by the basis.
            For half-integer numbers (e.g., F quanta), replace <j1> and <j2>
            in the template by <f1> and <f2>, these will be replaced by
            floating point numbers rounded to the first decimal.
        thresh : float
            Threshold for neglecting matrix elements.

    Returns:
        tens : field.CarTens
            Cartesian tensor
    """
    # read M and K tensors for different pairs of J quanta

    Jlist1 = states.Jlist1
    Jlist2 = states.Jlist2

    mmat = dict()
    kmat = dict()
    tens_cart = []
    tens_nirrep = None
    tens_ncart = None

    for J1 in Jlist1:
        for J2 in Jlist2:

            F1_str = str(round(J1,1))
            F2_str = str(round(J2,1))
            J1_str = str(int(round(J1,0)))
            J2_str = str(int(round(J2,0)))

            fname = re.sub(r"\<f1\>", F1_str, filename)
            fname = re.sub(r"\<f2\>", F2_str, fname)
            fname = re.sub(r"\<j1\>", J1_str, fname)
            fname = re.sub(r"\<j2\>", J2_str, fname)

            if not opersys.path.exists(fname):
                continue

            with open(fname, "r") as fl:

                iline = 0
                eof = False
                read_m = False
                read_k = False

                for line in fl:
                    strline = line.rstrip('\n')

                    if iline == 0:
                        if strline != "Start richmol format":
                            raise Exception(f"matrix elements file '{fname}' has bogus header '{strline}'")
                        iline+=1
                        continue

                    if strline == "End richmol format":
                        eof = True
                        break

                    if iline == 1:
                        w = strline.split()
                        tens_name = w[0]
                        nirrep = int(w[1])
                        ncart = int(w[2])
                        if tens_nirrep is not None and tens_nirrep != nirrep:
                            raise Exception(f"'nirrep' = {nirrep} read from file {fname} is different from " + \
                                f"the value {tens_nirrep} read from previous files") from None
                        if tens_ncart is not None and tens_ncart != ncart:
                            raise Exception(f"'ncart' = {ncart} read from file {fname} is different from " + \
                                f"the value {tens_ncart} read from previous files") from None
                        tens_nirrep = nirrep
                        tens_ncart = ncart
                        iline+=1
                        continue

                    if strline == "M-tensor":
                        read_m = True
                        read_k = False
                        mdata = dict()
                        mrow = dict()
                        mcol = dict()
                        iline+=1
                        continue

                    if strline == "K-tensor":
                        read_m = False
                        read_k = True
                        kdata = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                        krow = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                        kcol = {(sym1, sym2) : {} for sym1 in states.symlist1[J1] for sym2 in states.symlist2[J2]}
                        iline+=1
                        continue

                    if read_m is True and strline.split()[0] == "alpha":
                        w = strline.split()
                        icmplx = int(w[2])
                        cart = w[3].lower()
                        tens_cart = list(set(tens_cart + [cart]))
                        cmplx_fac = (1j, 1)[icmplx+1]
                        mdata[cart] = dict()
                        mrow[cart] = dict()
                        mcol[cart] = dict()
                        iline+=1
                        continue

                    if read_m is True:
                        w = strline.split()
                        m1 = float(w[0])
                        m2 = float(w[1])
                        try:
                            im1 = states.map_mstates[(J1, m1)]
                        except:
                            continue
                        try:
                            im2 = states.map_mstates[(J2, m2)]
                        except:
                            continue
                        mval = [ float(val) * cmplx_fac for val in w[2:] ]
                        if thresh is not None:
                            irreps = [i for i in range(nirrep) if abs(mval[i]) > thresh]
                        else:
                            irreps = [i for i in range(nirrep)]
                        for irrep in irreps:
                            try:
                                mrow[cart][irrep].append(im1)
                                mcol[cart][irrep].append(im2)
                                mdata[cart][irrep].append(mval[irrep])
                            except KeyError:
                                mrow[cart][irrep] = [im1]
                                mcol[cart][irrep] = [im2]
                                mdata[cart][irrep] = [mval[irrep]]

                    if read_k is True:
                        w = strline.split()
                        id1 = int(w[0])
                        id2 = int(w[1])
                        ideg1 = int(w[2])
                        ideg2 = int(w[3])
                        kval = [float(val) for val in w[4:]]
                        try:
                            istate1, sym1 = states.map_kstates[(J1, id1, ideg1)]
                        except:
                            continue
                        try:
                            istate2, sym2 = states.map_kstates[(J2, id2, ideg2)]
                        except:
                            continue
                        if thresh is not None:
                            irreps = [i for i in range(nirrep) if abs(kval[i]) > thresh]
                        else:
                            irreps = [i for i in range(nirrep)]
                        sym = (sym1, sym2)
                        for irrep in irreps:
                            try:
                                krow[sym][irrep].append(istate1)
                                kcol[sym][irrep].append(istate2)
                                kdata[sym][irrep].append(kval[irrep])
                            except IndexError:
                                krow[sym][irrep] = [istate1]
                                kcol[sym][irrep] = [istate2]
                                kdata[sym][irrep] = [kval[irrep]]

                    iline +=1

                if eof is False:
                    raise Exception(f"matrix-elements file '{fname}' has bogus footer '{strline}'")

            fshape = lambda sym: (states.dim_m1[J1][sym[0]], states.dim_m2[J2][sym[1]])
            mmat[(J1,J2)] = {sym : {cart : {irrep :
                             csr_matrix((mdata[cart][irrep], (mrow[cart][irrep], mcol[cart][irrep])), shape=fshape(sym))
                             for irrep in mdata[cart].keys()}
                             for cart in mdata.keys()}
                             for sym in kdata.keys()}

            fshape = lambda sym: (states.dim_k1[J1][sym[0]], states.dim_k2[J2][sym[1]])
            kmat[(J1,J2)] = {sym : {irrep :
                             csr_matrix((kdata[sym][irrep], (krow[sym][irrep], kcol[sym][irrep])), shape=fshape(sym))
                             for irrep in kdata[sym].keys()}
                             for sym in kdata.keys()}

            # delete empty entries in kmat
            symlist = list(kmat[(J1, J2)].keys())
            for sym in symlist:
                if all(mat.nnz == 0 for mat in kmat[(J1, J2)][sym].values()):
                    del kmat[(J1, J2)][sym]

            # delete empty entries in mmat
            symlist = list(mmat[(J1, J2)].keys())
            for sym in symlist:
                if all(mat.nnz == 0 for tmat in mmat[(J1, J2)][sym].values() for mat in tmat.values()):
                    del mmat[(J1, J2)][sym]

    # copy some attributes from states

    symlist1 = states.symlist1
    symlist2 = states.symlist2
    dim_m1 = states.dim_m1
    dim_m2 = states.dim_m2
    dim_k1 = states.dim_k1
    dim_k2 = states.dim_k2
    dim1 = states.dim1
    dim2 = states.dim2
    assign_m1 = states.assign_m1
    assign_m2 = states.assign_m2
    assign_k1 = states.assign_k1
    assign_k2 = states.assign_k2
    cart = tens_cart

    # irreps[(ncart, nirrep)]
    irreps = {(3,1) : [(1,-1), (1,0), (1,1)],                                               # rank-1 tensor
              (9,1) : [(2,-2), (2,-1), (2,0), (2,1), (2,2)],                                # traceless and symmetric rank-2 tensor
              (9,2) : [(0,0), (2,-2), (2,-1), (2,0), (2,1), (2,2)],                         # symmetric rank-2 tensor
              (9,3) : [(0,0), (1,-1), (1,0), (1,1), (2,-2), (2,-1), (2,0), (2,1), (2,2)]}   # non-symmetric rank-2 tensor

    # ranks[ncart]
    ranks = {3 : 1, 9 : 2}

    # infer list of spherical-tensor indices
    try:
        os = irreps[(tens_ncart, tens_nirrep)]
    except KeyError:
        raise ValueError(f"can't infer Cartesian tensor irreps from the number " + \
            f"of Cartesian components = {tens_ncart} and number of irreps = {tens_nirrep}") from None

    # infer rank
    try:
        rank = ranks[tens_ncart]
    except KeyError:
        raise ValueError(f"can't infer rank of Cartesian tensor from the number " + \
            f"of Cartesian components = {tens_ncart}") from None

    # initialize field.CarTens object

    loc = locals()
    tens = field.CarTens()
    for name in tens.attr_list:
        setattr(tens, name, loc[name])
    tens.check_attrs()

    return tens


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
