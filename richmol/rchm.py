class States():
    """ Basis of field-free molecular states """

    def __init__(self, fname, **kwargs):

        # read field-free states

        self.states, self.map_id_to_istate = self.read_states(fname, **kwargs)

        fmax = max(list(self.states.keys()))
        fmin = min(list(self.states.keys()))

        # generate list of M quanta

        if 'mlist' in kwargs:
            mlist = [round(m, 1) for m in kwargs['mlist']]
        else:
            if 'mmax' in kwargs:
                mmax = min([kwargs['mmax'], fmax])
                if mmax < kwargs['mmax']:
                    print(f"Psi: mmax is set to {mmax} which is maximum F in states file {fname}")
                if -mmax > fmin:
                    print(f"Psi: mmax is set to {mmax} and has a larger absolute value than \
                          fmin which is set to {fmin}")
            else:
                mmax = fmax
            if 'mmin' in kwargs:
                mmin = max([kwargs['mmin'], -fmax])
                if mmin > kwargs['mmin']:
                    print(f"Psi: mmin is set to {mmin} which is minus maximum F in states file {fname}")
                if mmin > fmin:
                    print(f"Psi: mmin is set to {mmin} and has a larger value than \
                          fmin which is set to {fmin}")
            else:
                mmin = -fmax
            if 'dm' in kwargs:
                dm = kwargs['dm']
            else:
                dm = 1
            m1 = round(mmin, 1)
            m2 = round(mmax, 1)
            d = round(dm)
            assert (m2 >= m1), f"mmin = {mmin} > mmax = {mmax}"
            assert (dm >=1 ),f "dm = {dm} < 1"
            self.mlist = [round(m,1) for m in np.linspace(m1,m2,int((m2-m1)/d)+1)]

        # generate basis: combinations of M and field-free state quanta for each F

        self.flist = [f for f in self.states.keys() if f>=min(map(abs, self.mlist))]

        self.quanta_m = {f:[m for m in self.mlist if abs(m)<=f] for f in self.flist}
        self.quanta_istate = {f:[state['istate'] for state in self.states[f]] for f in self.flist}
        self.dim_m = {f:len(self.quanta_m[f]) for f in self.flist}
        self.dim_istate = {f:len(self.quanta_istate[f]) for f in self.flist}

        self.quanta = {}
        self.energy = {}
        self.dim = {}
        self.f = []
        self.m = []
        self.istate = []
        for f in self.flist:
            self.quanta[f] = []
            enr = []
            for m in self.mlist:
                if abs(m)>f:
                    continue
                for state in self.states[f]:
                    istate = state['istate']
                    self.quanta[f].append([m,istate])
                    enr.append(state['enr'])
                    self.f.append(f)
                    self.m.append(m)
                    self.istate.append(istate)
                self.energy[f] = np.array(enr, dtype=np.float64)
            self.dim[f] = len(self.quanta[f])
            assert (self.dim[f] == self.dim_m[f] * self.dim_istate[f]), \
                    f"Basis dimension = {self.dim[f]} for J = {f} is not equal to the product " \
                    +f"of dimensions of m-basis = {self.dim_m[f]} and field-free basis = " \
                    +f"{self.dim_istate[f]}"

    def read_states(filename, **kwargs):
    """ Reads molecular field free energies and quantum numbers from Richmol states file """
    fl = open(filename, "r")

    # scan file for the number of states with different F quanta, max ID number, symmetry, etc.
    nstates = {}
    maxid = {}
    for line in fl:
        w = line.split()
        f = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2].upper()
        ndeg = int(w[3])
        enr = float(w[4])
        if 'emin' in kwargs and enr<kwargs['emin']:
            continue
        if 'emax' in kwargs and enr>kwargs['emax']:
            continue
        try:
            nstates[(f,sym)] += 1
        except:
            nstates[(f,sym)] = 1
        try:
            maxid[f] = max([id,maxid[f]])
        except:
            maxid[f] = id

    # create list of F quanta
    if 'flist' in kwargs:
        flist = [round(f,1) for f in kwargs['flist']]
    else:
        if 'fmax' in kwargs:
            fmax = min([ kwargs['fmax'], max([key[0] for key in nstates.keys()]) ])
            if fmax<kwargs['fmax']:
                print(f"read_states: fmax is set to {fmax} which is maximal F in states file {filename}")
        else:
            fmax = max([key[0] for key in nstates.keys()])
        if 'fmin' in kwargs:
            fmin = max([ kwargs['fmin'], min([key[0] for key in nstates.keys()]) ])
            if fmin>kwargs['fmin']:
                print(f"read_states: fmin is set to {fmin} which is minimal F in states file {filename}")
        else:
            fmin = min([key[0] for key in nstates.keys()])
        if 'df' in kwargs:
            df = kwargs['df']
        else:
            df = 1
        f1 = round(fmin,1)
        f2 = round(fmax,1)
        d = round(df)
        assert (f1>=0 and f2>=0),f"fmin={fmin} or fmax={fmax} is less than zero"
        assert (f2>=f1),f"fmin={fmin} > fmax={fmax}"
        assert (df>=1),f"df={df}<1"
        flist = [round(f,1) for f in np.linspace(f1,f2,int((f2-f1)/d)+1)]

    # create list of state symmetries
    if 'sym' in kwargs:
        sym_list = list( set(elem.upper() for elem in kwargs['sym']) & \
                         set([key[1] for key in nstates.keys()]) )
        bad_sym = list( set(elem.upper() for elem in kwargs['sym']) - \
                         set([key[1] for key in nstates.keys()]) )
        if len(bad_sym)>0:
            print(f"read_states: there are no states with symmetries {bad_sym} in states file {filename}")
    else:
        sym_list = set([key[1] for key in nstates.keys()])

    # read states

    states = {}
    map_id_to_istate = {}
    for f in flist:
        nst = sum([nstates[(f,sym)] for sym in sym_list if (f,sym) in nstates])
        if nst==0:
            continue
        states[f] = np.zeros( nst, dtype={'names':('f', 'id', 'ideg', 'istate', 'sym', 'enr', 'qstr'), \
            'formats':('f8', 'i8', 'i4', 'i8', 'U10', 'f8', 'U300')} )
        map_id_to_istate[f] = np.zeros( maxid[f]+1, dtype=np.int64 )
        map_id_to_istate[f][:] = -1

    fl.seek(0)

    nstates = {key[0]:0 for key in nstates}
    for line in fl:
        w = line.split()
        f = round(float(w[0]),1)
        id = np.int64(w[1])
        sym = w[2]
        ndeg = int(w[3])
        enr = float(w[4])
        qstr = ' '.join([w[i] for i in range(5,len(w))])
        if 'emin' in kwargs and enr<kwargs['emin']:
            continue
        if 'emax' in kwargs and enr>kwargs['emax']:
            continue
        if f not in flist:
            continue
        if sym.upper() not in sym_list:
            continue

        map_id_to_istate[f][id] = nstates[f]

        for ideg in range(ndeg):
            istate = nstates[f]
            states[f]['f'][istate] = f
            states[f]['id'][istate] = id
            states[f]['ideg'][istate] = ideg
            states[f]['sym'][istate] = sym
            states[f]['enr'][istate] = enr
            states[f]['qstr'][istate] = qstr
            states[f]['istate'][istate] = istate

            nstates[f] += 1

    fl.close()

    return states, map_id_to_istate


    @property
    def j_m_id(self):
        pass


    @j_m_id.setter
    def j_m_id(self, val):
        """Use this function to define the initial wavepacket coefficients, as
        Psi().j_m_id = (j, m, id, ideg, coef), where j, m, id, and ideg identify the stationary
        basis function and coef is the desired coefficient. Call it multiple times define the
        coefficient values for multiple basis functions.
        """
        try:
            f, m, id, ideg, coef = val
        except ValueError:
            raise ValueError(f"Pass an iterable with five items, i.e., j_m_id = (j, m, id, ideg, coef)")
        ff = round(float(f),1)
        mm = round(float(m),1)
        iid = int(id)
        iideg = int(ideg)
        try:
            x = self.flist.index(ff)
        except ValueError:
            raise ValueError(f"Input quantum number J = {ff} is not spanned by the basis") from None
        try:
            istate = self.map_id_to_istate[ff][iid]+iideg-1
        except IndexError:
            raise IndexError(f"Input set of quanta (id,ideg) = ({iid},{iideg}) for J = {ff} " \
                    + f"is not spanned by the basis") from None
        try:
            ibas = self.quanta[ff].index([mm,istate])
        except ValueError:
            raise ValueError(f"Input set of quanta (m,id,ideg,istate) = ({mm},{iid},{iideg},{istate}) " \
                    + f"for J = {ff} is not spanned by the basis") from None
        try:
            x = self.coefs
        except AttributeError:
            self.coefs = {f:np.zeros(len(self.quanta[f]), dtype=np.complex128) for f in self.flist}
        self.coefs[ff][ibas] = coef
