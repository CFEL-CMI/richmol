import h5py
import numpy as np
import re
import os
from scipy.sparse import coo_matrix, csr_matrix
import time
import datetime
from richmol.rot import molecule
from richmol.rot import labtens
from richmol.rot import mol_tens
from richmol.rot import solution
from richmol import field
import inspect


def J_group_key(J1, J2):
    return 'J:' + str(round(float(J1), 1)) + ',' + str(round(float(J2), 1))


def sym_group_key(sym1, sym2):
    return 'sym:' + str(sym1) + ',' + str(sym2)


def store(filename, obj, *args, **kwargs):
    """Stores objects in richmol hdf5 database file

    Args:
        filename : str
            Name of hdf5 file
        obj : object
            Object to store, currently one can store objects of type:
            rot.molecule.Molecule, rot.labtens.LabTensor, and field.CarTens
    """
    if isinstance(obj, molecule.Molecule):
        add_molecule(filename, obj, *args, **kwargs)
    elif isinstance(obj, (labtens.LabTensor, field.CarTens)):
        add_tensor(filename, obj, *args, **kwargs)
    else:
        raise TypeError(f"unsupported type: '{type(obj)}'") from None


def add_molecule(filename, mol, comment=None, replace=False):
    """Stores molecular data

    Args:
        filename : str
            Name of hdf5 file
        mol : Molecule
            Molecule object
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

        # store date/time
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        group.attrs['date'] = date.replace('\n','')

        # store user comment
        if comment is not None:
            group.attrs['comment'] = str.encode(" ".join(elem for elem in comment.split()))

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
                group.attrs['ABC'] = mol.ABC
            else:
                group.attrs['B'] = mol.B
        except AttributeError:
            pass

        # store tensors
        for tens in mol_tens._tensors.keys():
            try:
                group.attrs[tens] = getattr(mol, tens)
            except AttributeError:
                pass

        # store custom Hamiltonians and parameters
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


def get_molecule(filename):
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
    """Stores properties and matrix elements of Cartesian tensor operator

    Args:
        filename : str
            Name of hdf5 file
        tens : LabTensor
            Cartesian tensor object
        name : str
            Name of data group, by default, the name of variable passed as 'tens'
            will be used
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

        # store date/time
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        group.attrs['date'] = date.replace('\n','')

        # store user comment
        if comment is not None:
            group.attrs['comment'] = str.encode(" ".join(elem for elem in comment.split()))

        # store properties of tensor
        group.attrs['rank'] = tens.rank
        group.attrs['Us'] = tens.Us
        group.attrs['Ux'] = tens.Ux
        group.attrs['cart'] = tens.cart
        group.attrs['os'] = tens.os
        # store molecular frame tensor elements
        group.attrs['tens_flat'] = [elem for elem in tens.tens_flat]

        # store matrix elements

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


def inspect_file(filename):
    """Returns information about all data groups stored in file

    Arg:
        filename : str
            Name of hdf5 file

    Returns:
        datagroup : dict
            Dictionary containing attributes of different data groups stored in file.
            For tensor data groups, additionally, a list of pairs of coupled
            J quanta is loaded into the attribute 'Jpairs', list of pairs
            of coupled symmetries (for different pairs of J quanta)
            is loaded into the attribute 'symlist'.
    """
    J_key_re = re.sub(r'1.0', '\d+\.\d+', J_group_key(1, 1))
    sym_key_re = re.sub(r'A', '\w+', sym_group_key('A', 'A'))

    datagroup = dict()

    with h5py.File(filename, 'a') as fl:
        for name in fl.keys():

            # read data group attributes
            datagroup[name] = type(name, (object,), {key:val for key,val in fl[name].attrs.items()})

            # for tensor data groups ...
            #   read list of pairs of J quanta coupled in tensor matrix elements
            #   read list of pairs of symmetries for each pair of J quanta
            Jlist = []
            symlist = dict()
            if isinstance(fl[name], h5py.Group):
                for Jkey in fl[name].keys():

                    if re.match(J_key_re, Jkey):
                        Jpair = re.findall(f'\d+.\d+', Jkey)
                        J1, J2 = (round(float(elem), 1) for elem in Jpair)
                        Jlist.append((J1,J2))

                        for symkey in fl[name][Jkey]:

                            if re.match(sym_key_re, symkey):
                                sympair = re.findall(f'\w+', symkey)
                                _, sym1, sym2 = sympair
                                try:
                                    symlist[(J1, J2)].append((sym1, sym2))
                                except KeyError:
                                    symlist[(J1, J2)] = [(sym1, sym2)]

                datagroup[name].Jpairs = Jlist
                datagroup[name].sympairs = symlist

    return datagroup


def get_tensor(filename, name):
    """Reads properties and matrix elements of Cartesian tensor operator

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

    with h5py.File(filename, 'a') as fl:
        try:
            group = fl[name]
        except KeyError:
            raise KeyError(f"file '{filename}' has no tensor dataset '{name}'") from None

        # read dataset attributes
        tens = type(name, (object,), {key:val for key,val in group.attrs.items()})

        # read tensor matrix elements
    
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

    return tens


def old_to_new_richmol(h5_file, states_file, tens_file=None, replace=False, store_states=True, \
                       me_tol=1e-40, **kwargs):
    """Converts richmol old formatted text file format into to new hdf5 file format.

    Args:
        h5_fname : str
            Name of new richmol hdf5 file.
        states_fname : str
            Name of old richmol states file.
        tens_fname : str
            Template for generating names of old richmol tensor matrix elements files.
            For example, for filename="matelem_alpha_j<j1>_j<j2>.rchm" following files will be
            searched: matelem_alpha_j0_j0.rchm, matelem_alpha_j0_j1.rchm,
            matelem_alpha_j0_j2.rchm and so on, where <j1> and <j2> will be replaced by integer
            numbers running through all J quanta spanned by all states listed in file states_fname.
            NOTE: in old richmol format, j1 and j2 are treated as ket and bra state quanta, respectively.
            For half-integer numbers (e.g., F quanta), substitute <j1> and <j2> in the template
            tens_fname by <f1> and <f2>, which will then be replaced by floating point numbers
            rounded to the first decimal.
        replace : bool
            Set True to replace existing hdf5 file.
        store_states : bool
            Set True to write richmol states data into hdf5 file, or False if you call this function
            multiple times to convert different tensors and don't want each time to rewrite the states
            data. The states file however need to be loaded for every new tensor.
        me_tol : float
            Threshold for neglecting matrix elements.
    Kwargs:
        descr : str or list
            Description of richmol data. Old richmol file format does not contain description
            of the data, this can be added into hdf5 file by passing a string or list of strings
            in descr variable. The description can be added only if store_states=True.
        tens_descr : str or list
            Description of tensor.
        tens_units : str
            Tensor units.
        enr_units : str
            Energy units.
        tens_name : str
            If provided, the name of tensor read from richmol file will be replaced with tens_name.
            Tensor name is used to generate data group name for matrix elements of tensor in hdf5 file.
            For state energies, the data group name is 'h0'.
    """
    if replace == True:
        store_states_ = True
    else:
        store_states_ = store_states

    try:
        descr = kwargs['descr']
    except KeyError:
        descr = None

    try:
        tens_descr = kwargs['tens_descr']
    except KeyError:
        tens_descr = None

    try:
        tens_units = kwargs['tens_units']
    except KeyError:
        tens_units = None

    try:
        enr_units = kwargs['enr_units']
    except KeyError:
        enr_units = None

    try:
        tens_name = kwargs['tens_name']
    except KeyError:
        tens_name = None

    # read states data
    print(f"Read states data from richmol formatted text file {states_file}")
    fl = open(states_file, "r")
    maxid = {}
    for line in fl:
        w = line.split()
        J = round(float(w[0]),1)
        id = np.int64(w[1])
        try:
            maxid[J] = max(maxid[J], id)
        except KeyError:
            maxid[J] = id
    fl.seek(0)
    map_id_to_istate = {J : np.zeros(maxid[J]+1, dtype=np.int64) for J in maxid.keys()}
    for J, elem in map_id_to_istate.items():
        elem[:] = -1
    states = {}
    nstates = {J : 0 for J in maxid.keys()}
    for line in fl:
        w = line.split()
        J = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2]
        ndeg = int(w[3])
        enr = float(w[4])
        qstr = ' '.join([w[i] for i in range(5,len(w))])
        try:
            map_id_to_istate[J][id] = nstates[J]
        except KeyError:
            map_id_to_istate[J][id] = 0
        try:
            x = states[J][0]
        except (IndexError, KeyError):
            states[J] = []
        for ideg in range(ndeg):
            states[J].append((nstates[J], sym, ideg, enr, qstr))
            nstates[J] += 1
    fl.close()

    # store states data
    if store_states_ == True:
        print(f"Write states data into richmol hdf5 file {h5_file}, replace={replace}")
        for iJ, J in enumerate(states.keys()):
            id = [elem[0] for elem in states[J]]
            sym = [elem[1] for elem in states[J]]
            ideg = [elem[2] for elem in states[J]]
            enr = [elem[3] for elem in states[J]]
            qstr = [elem[4] for elem in states[J]]
            store(h5_file, 'h0', J, J, enr=enr, id=id, sym=sym, ideg=ideg, assign=qstr, \
                  replace=(replace if iJ == 0 else False), descr=descr, units=enr_units)

    # read matrix elements data and store into hdf5 file
    print(f"Convert richmol matrix elements data {tens_file} --> {h5_file}, tol={me_tol}")
    for J1 in states.keys():
        for J2 in states.keys():

            F1_str = str(round(J1,1))
            F2_str = str(round(J2,1))
            J1_str = str(int(round(J1,0)))
            J2_str = str(int(round(J2,0)))

            fname = re.sub(r"\<f1\>", F1_str, tens_file)
            fname = re.sub(r"\<f2\>", F2_str, fname)
            fname = re.sub(r"\<j1\>", J1_str, fname)
            fname = re.sub(r"\<j2\>", J2_str, fname)

            if not os.path.exists(fname):
                continue

            print(f"Read matrix elements from richmol formatted text file {fname}")
            fl = open(fname, "r")

            iline = 0
            eof = False
            read_m = False
            read_k = False
            icart = None

            for line in fl:
                strline = line.rstrip('\n')

                if iline==0:
                    if strline!="Start richmol format":
                        raise RuntimeError(f"Matrix-elements file '{fname}' has bogus header = '{strline}'")
                    iline+=1
                    continue

                if strline == "End richmol format":
                    eof = True
                    break

                if iline==1:
                    w = strline.split()
                    if tens_name is None:
                        tens_name = w[0]
                    nomega = int(w[1])
                    ncart = int(w[2])
                    iline+=1
                    continue

                if strline=="M-tensor":
                    read_m = True
                    read_k = False
                    mvec = {}
                    im1 = {}
                    im2 = {}
                    iline+=1
                    continue

                if strline=="K-tensor":
                    read_m = False
                    read_k = True
                    kvec = [[] for i in range(nomega)]
                    ik1 = [[] for i in range(nomega)]
                    ik2 = [[] for i in range(nomega)]
                    iline+=1
                    continue

                if read_m is True and strline.split()[0]=="alpha":
                    w = strline.split()
                    icart = int(w[1])
                    icmplx = int(w[2])
                    scart = w[3].lower()
                    cmplx_fac = (1j, 1)[icmplx+1]
                    mvec[scart] = [[] for i in range(nomega)]
                    im1[scart] = [[] for i in range(nomega)]
                    im2[scart] = [[] for i in range(nomega)]
                    iline+=1
                    continue

                if read_m is True:
                    w = strline.split()
                    m1 = float(w[0])
                    m2 = float(w[1])
                    i1, i2 = int(m1 + J1), int(m2 + J2)
                    mval = [ float(val) * cmplx_fac for val in w[2:] ]
                    ind_omega = [i for i in range(nomega) if abs(mval[i]) > me_tol]
                    for iomega in ind_omega:
                        im1[scart][iomega].append(i1)
                        im2[scart][iomega].append(i2)
                        mvec[scart][iomega].append(mval[iomega])

                if read_k is True:
                    w = strline.split()
                    id1 = int(w[0])
                    id2 = int(w[1])
                    ideg1 = int(w[2])
                    ideg2 = int(w[3])
                    kval = [float(val) for val in w[4:]]
                    istate1 = map_id_to_istate[J1][id1] + ideg1 - 1
                    istate2 = map_id_to_istate[J2][id2] + ideg2 - 1
                    ind_omega = [i for i in range(nomega) if abs(kval[i]) > me_tol]
                    for iomega in ind_omega:
                        ik1[iomega].append(istate1)
                        ik2[iomega].append(istate2)
                        kvec[iomega].append(kval[iomega])

                iline +=1
            fl.close()

            if eof is False:
                raise RuntimeError(f"Matrix-elements file '{fname}' has bogus footer = '{strline}'")

            print(f"Write matrix elements into richmol hdf5 file {h5_file}, me_tol = {me_tol}")

            # write M-matrix into hdf5 file
            for cart in mvec.keys():
                print(f"    M-tensor's {cart} component, irreps = {ind_omega}")
                ind_omega = [iomega for iomega in range(nomega) if len(mvec[cart][iomega]) > 0]
                mat = [ coo_matrix((mvec[cart][iomega], (im2[cart][iomega], im1[cart][iomega])), \
                            shape=(int(2*J2+1), int(2*J1+1)) ) for iomega in ind_omega ]
                store(h5_file, tens_name, J2, J1, thresh=me_tol, irreps=ind_omega, cart=cart, mmat=mat)

            # write K-matrix into hdf5 file
            print(f"    K-tensor, irreps = {ind_omega}")
            ind_omega = [iomega for iomega in range(nomega) if len(kvec[iomega]) > 0]
            mat = [ coo_matrix((kvec[iomega], (ik2[iomega], ik1[iomega])), shape=(nstates[J2], nstates[J1]) ) \
                    for iomega in ind_omega ]
            store(h5_file, tens_name, J2, J1, thresh=me_tol, irreps=ind_omega, kmat=mat, \
                  tens_descr=tens_descr, units=tens_units)


def retrieve_name(var):
    """ Gets the name of var. Does it from the out most frame inner-wards """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
