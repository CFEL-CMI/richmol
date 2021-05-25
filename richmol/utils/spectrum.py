from richmol.convert_units import (
    Debye_to_au, Debye_to_si,
    Debye_to_sqrt_erg_x_sqrt_cm3,
    Buckingham_to_au,
    Buckingham_to_si,
    Buckingham_to_sqrt_erg_x_sqrt_cm5
)
from richmol.field import CarTens
import numpy as np
import scipy as sp
import scipy.constants as const
import h5py
from mpi4py import MPI
import matplotlib.pyplot as plt


class field_free_spectrum():
    """ Class for field-free spectrum

        Attrs:
            filename : str
                Name of the HDF5 file from which tensor data is loaded.
                Alternatively, one can load tensor from the old-format ASCII
                files, by providing in `filename` the name of the richmol
                states file and in `matelem` a template for generating the
                names of the richmol matrix elements files.
            matelem : str
                In the old-format, matrix elements of Cartesian tensors for
                different values of bra and ket J (or F) quanta are stored in
                separate files.
            out_f : str
                Name of the HDF5 file to which results are written.
            type : str
                The type of coupling moment ('elec', 'magn'). Set to 'elec' by
                default.
            order : str
                The order of coupling moment. If `matelem` is given, order will
                be deduced from `matelem`. Else must be specified here ('dip',
                'quad').
            j_min : int, float
                The minimum value of J (or F). Set to 0.0 (for integer) or 0.5
                (for half-integer) by default.
            j_max : int, float
                The maximum value of J (or F). Set to 1.0 (for 'dip') or 2.0
                (for 'quad') for integer by default. Set to 0.5 (for 'dip') or
                1.5 (for 'quad') for half-integer by default.
            e_max : int, float
                The maximum value of energy. Set to 1.0e9 by default.
            units : str
                The units of coupling moment. Set to 'Debye' by default.
            linestr_thresh : int, float
                The linestrength threshold below which to neglect transitions.
                Set to 0 by default.
            abun : int, float
                The natural terrestrial isotopic abundance. Set to 1.0 by
                default.
            temp : int, float
                The temperature. Set to 296.0 by default.
            part_sum : int, float
                The total internal partition sum. Set to 1.0 by default.
            filters: list
                User defined filtering functions for transitions. The folowing
                keyword arguments are passed into the filter functions:
                    sym : tuple
                        Symmetries of bra and ket state.
                    qstr : tuple
                        Qstrs of bra and ket state.
                Set to [def all_(**kwargs): return True]
            abs_intens_thresh : int, float
                The absorption intensity threshold below which to neglect
                transitions. Set to 0 by default.

        Methods:
            __init__(filename, matelem=None, **kwargs)
                Initializes field-free spectrum object, computes assignments.
            linestr(**kwargs)
                Computes linestrengths.
            abs_intens(**kwargs)
                Computes, filters and plots  absorption intensities.
            totxt(**kwargs)
                Writes results into ASCII file.
    """

    def __init__(self, filename, matelem=None, **kwargs):
        """ Initializes a field-free spectrum object.

            Args:
                filename : str
                    Name of the HDF5 file from which tensor data is loaded.
                    Alternatively, one can load tensor from the old-format
                    ASCII files, by providing in `filename` the name of the
                    richmol states file and in `matelem` a template for
                    generating the names of the richmol matrix elements files.
                matelem : str
                    In the old-format, matrix elements of Cartesian tensors for
                    different values of bra and ket J (or F) quanta are stored
                    in separate files.

            Kwargs:
                type : str
                    The type of coupling moment ('elec', 'magn'). Set to 'elec'
                    by default.
                order : str
                    The order of coupling moment. If `matelem` is given, order
                    will be deduced from `matelem`. Else must be specified here
                    ('dip', 'quad').
                j_max : int, float
                    The maximum value of J (or F). If `j_min` is 0.0, set to
                    1.0 (for 'dip') or 2.0 (for 'quad') by default. If `j_min`
                    is 0.5, set to 0.5 (for 'dip') or 1.5 (for 'quad') by
                    default.
                e_max : int, float
                    The maximum value of energy. Set to 1.0e9 by default.
                units : str
                    The units of coupling moment. Set to 'Debye' by default.
        """

        # set main filename
        self.filename = filename

        # set matelem filename
        self.matelem = matelem

        # set coupling moment type
        if 'type' in kwargs:
            assert (kwargs['type'] in ['elec', 'magn']), \
                f"'type' unknown '{kwargs['type']}'" \
                    + f"(must be 'elec', 'magn')"
            self.type = kwargs['type']
        else:
            self.type = 'elec'

        # set coupling moment order
        if matelem is not None:
            assert ('_MU' in matelem or '_QUAD' in matelem), \
                f"'order' not found in 'matelem' = '{matelem}'" \
                    + f"(must contain '_MU', '_QUAD')"
            if '_MU' in matelem:
                self.order = 'dip'
            else:
                self.order = 'quad'
        else:
            assert ('order' in kwargs), \
                f"'order' not found in keyword arguments" \
                    + f"(set to 'dip', 'quad')"
            assert (kwargs['order'] in ['dip', 'quad']), \
                f"'order' unknown" \
                    + f"(set to 'dip', 'quad')"
            self.order = kwargs['order']

        # set minimum and maximum J
        if 'j_max' in kwargs:
            assert (type(kwargs['j_max']) in [int, float]), \
                f"'j_max' has bad type: '{type(kwargs['j_max'])}'" \
                    + f"(must be 'int' or 'float')"
            assert (abs(kwargs['j_max'] % 0.5) in [0.0, 0.5]), \
                f"'j_max' has bad value: '{kwargs['j_max']}'" \
                    + f"(must be 'int', 'half-int')"
            if abs(kwargs['j_max'] % 1.0) == 0.0:
                self.j_min = 0.0
                j_max_min = {'dip' : 1.0, 'quad' : 2.0}[self.order]
                assert (kwargs['j_max'] >= j_max_min), \
                   f"'j_max' has bad value: '{kwargs['j_max']}'" \
                        + f"(must be >= '{j_max_min}')"
            else:
                self.j_min = 0.5
                j_max_min = {'dip' : 0.5, 'quad' : 1.5}[self.order]
                assert (kwargs['j_max'] >= j_max_min), \
                   f"'j_max' has bad value: '{kwargs['j_max']}'" \
                        + f"(must be >= '{j_max_min}')"
            self.j_max = round(float(kwargs['j_max']), 1)
        else:
            self.j_min = 0.0
            self.j_max = {'dip' : 1.0, 'quad' : 2.0}[self.order]

        # set maximum enr
        if 'e_max' in kwargs:
            assert ((type(kwargs['e_max']) in [int, float])), \
                f"'e_max' has bad type '{type(kwargs['e_max'])}'" \
                    + f"(must be 'int' or 'float')"
            assert (kwargs['e_max'] > 0), \
                f"'e_max' has bad value '{kwargs['e_max']}'" \
                    + f"(must be > '0')"
            self.e_max = kwargs['e_max']
        else:
            self.e_max = 1.0e9

        # set physical units of moment
        if 'units' in kwargs:
            if self.order == 'dip':
                assert (kwargs['units'] in ['Debye', 'a.u.', 'S.I.']), \
                    f"'units' unknown: '{kwargs['units']}'" \
                        + f"(must be 'Debye', 'a.u.', 'S.I.')"
            else:
                assert (kwargs['units'] in ['Buckingham', 'a.u.', 'S.I.']), \
                    f"'units' unknown: '{kwargs['units']}'" \
                        + f"(must be 'Buckingham', 'a.u.', 'S.I.')"
            self.units = kwargs['units']
        else:
            self.units = 'Debye'

        # construct output filename
        self.out_f = '{}_jmax{}.hdf5'.format(self.order, self.j_max)


    def linestr(self, **kwargs):
        """ Wrapper for computing assignments and linestrengths.

            Kwargs:
                thresh : float
                    The linestrength threshold below which to neglect
                    transitions. Set to 0 by default.
        """

        # set linestrength threshold
        if 'thresh' in kwargs:
            assert (type(kwargs['thresh']) in [int, float]), \
                f"'thresh' has bad type: '{type(kwargs['thresh'])}'" \
                   + f"(must be 'int', 'float')"
            assert (kwargs['thresh'] >= 0), \
                f"'thresh' has bad value: '{kwargs['thresh']}'" \
                    + f"(must be >= '0')"
            self.linestr_thresh = kwargs['thresh']
        else:
            self.linestr_thresh = 0.0

        # assignments
        self._assign()

        # linestrengths
        self._linestr()


    def _assign(self):
        """ Assigns properties (energy, symmetry, nuclear spin, ...) to
              field-free states.

            Results are saved into 'self.out_f'. Assignments are saved as
              structured arrays into datasets '<J>' of the group 'assign'. Such
              an array has the fields 'enr' (energy), 'sym' (symmetry), 'ns'
              (nuclear spin) and 'assign' (other assignments).
        """

        # initialize MPI variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # input / output file
        f = h5py.File(self.out_f, 'w', driver='mpio', comm=MPI.COMM_WORLD)

        # filters ket states w.r.t. J, M and energy
        def filt_ket(**kwargs):
            pass_J, pass_M, pass_enr = True, True, True
            if 'J' in kwargs:
                pass_J = kwargs['J'] == j
            if 'm' in kwargs:
                pass_M = kwargs['m'] == ' 0.0'
            if 'enr' in kwargs:
                pass_enr = kwargs['enr'] <= self.e_max
            return pass_J and pass_M and pass_enr

        # datatype of assignment
        assign_dtype = [('enr', 'f8'), ('sym', 'S2'), ('qstr', '<S50')]

        # assignments
        assigns, shapes = {}, {}
        j = self.j_min + rank
        while j <= self.j_max:

            # field-free states
            free_states = CarTens(
                self.filename, 
                bra = lambda **kwargs : False, 
                ket = filt_ket
            )

            # assignment
            _, assign_ket = free_states.assign(form='full')
            assign = []
            for ind in range(len(assign_ket['J'])):
                 qstr, enr = assign_ket['k'][ind]
                 sym = assign_ket['sym'][ind]
                 assign.append((enr, sym, qstr))

            assigns[j] = np.array(assign, dtype=assign_dtype)
            shapes[j] = assigns[j].shape

            j += world_size

        # broadcast shapes...
        shapes_ = {}
        for rank_ in range(world_size):
            if rank_ == rank:
                data = shapes
            else:
                data = None
            shapes_.update(comm.bcast(data, root=rank_))

        # ...to collectively create datasets...
        f.create_group('assign')
        for j in list(shapes_.keys()):
            f['assign'].create_dataset(str(j), shapes_[j], dtype=assign_dtype)

        # ...and write into datasets in parallel
        for j in list(assigns.keys()):
            f['assign'][str(j)][:] = assigns[j]

        # print to console
        if rank == 0:
            print('\n  COMPUTED ASSIGNMENTS ...\n')
            for attr in ['filename', 'j_min', 'j_max', 'e_max']:
                print('      {} = {}'.format(attr, getattr(self, attr)))

        comm.Barrier()


    def _linestr(self):
        """ Computes linestrengths of transitions from lower field-free states
              to higher field-free states.

            Results are saved into 'self.out_f'. Linestrengths are saved as
              structured arrays into datasets '(<J_bra>, <J_ket>)' of the group
              'spec'. Such an array has the fields 'bra ind' (assignment index
              of bra state), 'ket ind' (assignment index of ket state), 'freq'
              (frequency of transition), 'linestr' (linestrength of
              transition), 'intens' (absorption intensity of transition).
        """

        # initialize MPI variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # input / output file
        f = h5py.File(self.out_f, 'a', driver='mpio', comm=MPI.COMM_WORLD)

        # filters states w.r.t. J, M and energy
        def filt(j):
            def filter_(**kwargs):
                pass_J, pass_enr = True, True
                if 'J' in kwargs:
                    pass_J = kwargs['J'] == j
                if 'enr' in kwargs:
                    pass_enr = kwargs['enr'] <= self.e_max
                return pass_J and pass_enr
            return filter_

        # datatype of spectrum
        spec_dtype = [
            ('bra ind', 'i4'),
            ('ket ind', 'i4'),
            ('freq', 'f8'),
            ('linestr', 'f8'),
            ('intens', 'f8')
        ]

        # linestrengths
        specs, shapes = {}, {}
        j_bra = self.j_min + rank
        while j_bra <= self.j_max:

            for j_ket in np.arange(self.j_min, self.j_max + 1):
                # selection rules
                if self.order == 'dip':
                    if abs(j_ket - j_bra) > 1 or (j_ket + j_bra) < 1: continue
                elif self.order == 'quad':
                    if abs(j_ket - j_bra) > 2 or (j_ket + j_bra) < 2: continue

                # Cartesian tensor operator
                moment = CarTens(
                    self.filename,
                    matelem = self.matelem,
                    bra = filt(j_bra),
                    ket = filt(j_ket)
                )

                # linestrength
                linestr = []
                for cart in moment.cart:

                    # block form
                    moment_blocks = moment.tomat(cart=cart)[(j_bra, j_ket)]

                    linestr_blocks = {}
                    for symsym, block in moment_blocks.items():

                        dim_bra, dim_ket = block.shape
                        dim_m_bra = int(dim_bra / (2 * j_bra + 1))
                        dim_m_ket = int(dim_ket / (2 * j_ket + 1))

                        # linestrength of block
                        linestren_ = block.multiply(block.conjugate())

                        # sum up rows of different M_bra
                        linestren_ = sum(
                            [ linestren_[i * dim_m_bra : (i + 1) * dim_m_bra, :] 
                                for i in range(int(2 * j_bra + 1)) ]
                        ).tocsc()

                        # sum up cols of different M_ket
                        linestren_ = sum(
                            [ linestren_[:, i * dim_m_ket : (i + 1) * dim_m_ket] 
                                for i in range(int(2 * j_ket + 1)) ]
                        )

                        linestr_blocks[symsym] = linestren_

                    # full form
                    linestr.append(
                        sp.sparse.bmat(
                            [
                                [
                                    linestr_blocks[(sym1, sym2)]
                                      if (sym1, sym2) in linestr_blocks.keys()
                                      else None
                                      for sym2 in moment.symlist2[j_ket]
                                ]
                              for sym1 in moment.symlist1[j_bra]
                            ]
                        )
                    )

                linestr = sum(linestr).tocoo()

                # structured spectrum array
                spec = np.empty((linestr.count_nonzero(), ), dtype=spec_dtype)
                spec['bra ind'] = linestr.row
                spec['ket ind'] = linestr.col
                spec['linestr'] = linestr.data.real
            
                # frequencies
                enr_bra = f['assign'][str(j_bra)][:]['enr'][spec['bra ind']]
                enr_ket = f['assign'][str(j_ket)][:]['enr'][spec['ket ind']]
                spec['freq'] = abs(enr_bra - enr_ket)

                # transitions from lower to higher state
                spec = spec[np.argwhere(enr_ket - enr_bra < 0).flatten()]

                # threshold
                if self.linestr_thresh > 0:
                    spec = spec[
                        np.argwhere(
                            spec['linestr'] > self.linestr_thresh
                        ).flatten()
                    ]

                if spec.size > 0:
                    specs[(j_bra, j_ket)] = spec
                    shapes[(j_bra, j_ket)] = specs[(j_bra, j_ket)].shape

            j_bra += world_size

        # broadcast shapes...
        shapes_ = {}
        for rank_ in range(world_size):
            if rank_ == rank:
                data = shapes
            else:
                data = None
            data = comm.bcast(data, root=rank_)
            shapes_.update(data)
    
        # ...to collectively create datasets...
        f.create_group('spec')
        for jj in list(shapes_.keys()):
            f['spec'].create_dataset(str(jj), shapes_[jj], dtype=spec_dtype)

        # ...and write into datasets in parallel
        for jj in list(specs.keys()):
            f['spec'][str(jj)][:] = specs[jj]

        f.close()

        # print to console
        if rank == 0:
            print('\n  COMPUTED LINESTRENGTHS ...\n')
            if self.matelem is None:
                attr_list = ['filename', 'type', 'order', 'linestr_thresh']
            else:
                attr_list = ['matelem', 'type',  'order', 'linestr_thresh']
            for attr in attr_list:
                print('      {} = {}'.format(attr, getattr(self, attr))) 

        comm.Barrier()


    def abs_intens(self, temp, part_sum, **kwargs):
        """ Wrapper for computing, filtering and plotting absorption
              intensities.

            Args:
                temp : int, float
                    The temperature.
                part_sum : int, float
                    The total internal partition sum.

            Kwargs:
                filters: list
                    User defined filtering functions for transitions (see
                    `filters` in kwargs of :py:class:`field_free_spectrum`).
                thresh : int, float
                    The absorption intensity threshold below which to neglect
                    transitions. Set to 0 by default.
        """

        # set temperature
        assert (type(temp) in [int, float]), \
            f"'temp' has bad type: '{type(temp)}'" \
                + f"(must be 'int', 'float')"
        assert (temp > 0), \
            f"'temp' has bad value: '{temp}'" \
                + f"(must be > '0')"
        self.temp = temp

        # set internal partition sum
        assert (type(part_sum) in [int, float]), \
            f"'part_sum' has bad type: '{type(part_sum)}'" \
                + f"(must be 'int', 'float')"
        assert (part_sum > 0), \
            f"'part_sum' has bad value '{part_sum}'" \
                + f"(must be > '0')"
        self.part_sum = part_sum

        # set natural terrestrial isotopic abundance
        if 'abun' in kwargs:
            assert (type(kwargs['abun']) in [int, float]), \
                f"'abun' has bad type: '{type(kwargs['abun'])}'" \
                    + f"(must be 'int', 'float')"
            assert (kwargs['abun'] > 0), \
                f"'abun' has bad value '{kwargs['abun']}'" \
                     + f"(must be > 0)"
            self.abun = kwargs['abun']
        else:
            self.abun = 1.0

        # set absorption intensity threshold
        if 'thresh' in kwargs:
            assert (type(kwargs['thresh']) in [int, float]), \
                f"'thresh' has bad type: '{type(kwargs['thresh'])}'" \
                    + f"(must be 'int', 'float')"
            assert (kwargs['thresh'] > 0), \
                f"'thresh' has bad value '{kwargs['thresh']}'" \
                     + f"(must be > 0)"
            self.abs_intens_thresh = kwargs['thresh']
        else:
            self.abs_intens_thresh = 0.0

        # set transition filters
        if 'filters' in kwargs:
            self.filters = kwargs['filters']
        else:
            def all_(**kwargs): return True
            self.filters = [all_]

        # absorption intensities
        self._abs_intens()

        # filter absorption intensities
        self._filter()

        # plot filtered absorption intensities
        self._plot()


    def _abs_intens(self):
        """ Computes absorption intensities of transitions.

            Results are saved into 'out_f'. Intensities are saved as structured
              arrays into datasets '(<F_bra>, <F_ket>)' of the group
              'spectrum'. See '_linestrengths(...)' for fields of such an array.
        """

        # initialize MPI variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # input / output file
        f = h5py.File(self.out_f, 'a', driver='mpio', comm=MPI.COMM_WORLD)

        # linestrength conversion factor
        conversion_factor = {
            'elec' : {
                'dip' : { # (..) -> (Debye^2) -> (erg*cm^3)
                    'Debye' : Debye_to_sqrt_erg_x_sqrt_cm3()**2,
                    'a.u.' : ( Debye_to_sqrt_erg_x_sqrt_cm3() \
                        / Debye_to_au() )**2,
                    'S.I.' : ( Debye_to_sqrt_erg_x_sqrt_cm3() \
                        / Debye_to_si() )**2
                },
                'quad' : { # (..) -> (Buckingham^2) -> (erg*cm^5)
                    'Buckingham' : Buckingham_to_sqrt_erg_x_sqrt_cm5()**2,
                    'a.u.' : ( Buckingham_to_sqrt_erg_x_sqrt_cm5() \
                        / Buckingham_to_au() )**2,
                    'S.I.' : ( Buckingham_to_sqrt_erg_x_sqrt_cm5() \
                        / Buckingham_to_si() )**2
                }
            },
            'magn' : {
                'dip' : { # (..) -> (?^2) -> (erg*cm^5/s^2)
                    '?' : 1.0, # TODO
                    'a.u.' : 1.0, # TODO
                    'S.I.' : 1.0 # TODO
                }
            }
        }[self.type][self.order][self.units]

        # absorption intensity coefficient
        c_1 = const.h * 1e2 * const.c / (const.k * self.temp) # (cm)
        c_2 = 8 * sp.pi**3 / (3 * 1.0e9  * const.h * const.c    * self.part_sum) # (1/erg/cm)
        c_3 = 4 * sp.pi**5 / (5 * 1.0e9  * const.h * const.c    * self.part_sum) # (1/erg/cm)
        c_4 = 8 * sp.pi**4 / (3 * 1.0e13 * const.h * const.c**3 * self.part_sum) # (s^2/erg/cm^3)

        # absorption intensity function
        abs_intens_func = {
            'elec' : {
                'dip' :
                    lambda R, E_low, v : c_2 * self.abun * v * \
                        conversion_factor * R * \
                        np.exp(-c_1 * E_low) * (1 - np.exp(-c_1 * v)),
                'quad' :
                    lambda R, E_low, v : c_3 * self.abun * v**3 * \
                        conversion_factor * R * \
                        np.exp(-c_1 * E_low) * (1 - np.exp(-c_1 * v))
            },
            'magn' : {
                'dip' :
                    lambda R, E_low, v : c_4 * self.abun * v * \
                        conversion_factor * R *  \
                        np.exp(-c_1 * E_low) * (1 - np.exp(-c_1 * v))
            }
        }[self.type][self.order]

        # absorption intensities
        jj_list = list(f['spec'].keys())
        ind = rank
        while ind < len(jj_list):
            j_bra, j_ket = [j for j in jj_list[ind][1: -1].split(', ')]

            # linestrengths and assignments
            spec = f['spec'][jj_list[ind]][:]
            assign_ket = f['assign'][j_ket][:]

            # absorption intensities
            spec['intens'] = abs_intens_func(
                spec['linestr'],
                assign_ket['enr'][spec['ket ind']],
                spec['freq']
            )

            f['spec'][jj_list[ind]][:] = spec

            ind += world_size

        f.close()

        # print to console
        if rank == 0:
            print('\n  COMPUTED ABSORPTION INTENSITIES ...\n')
            for attr in ['type', 'order', 'units', 'abun', 'temp', 'part_sum']:
                print('      {} = {}'.format(attr, getattr(self, attr)))

        comm.Barrier()


    def _filter(self):
        """ Filters out absorption intensities.

            Results are saved into 'out_f'. Filtered intensities are saved as one
              contiguous structured array into the dataset 'raw' of the group
              '<transition_type>'. This array has the fields 'F ket' (value of F
              quantum number of ket state), 'F bra' (value of F quantum number of
              bra state), 'ket index' (assignment index of ket state) 'bra index'
              (assignment index of bra state), 'frequency' (frequency of transition)
              and 'intensity' (absorption intensity of transition).
        """

        # initialize MPI variables
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # input / output file
        f = h5py.File(self.out_f, 'a', driver='mpio', comm=MPI.COMM_WORLD)
        for filt in self.filters:
            if filt.__name__ in list(f.keys()):
                del f[filt.__name__]

        # datatype of absorption intensities
        intens_dtype = [
            ('J ket', 'f4'),
            ('J bra', 'f4'),
            ('ket ind', 'i4'),
            ('bra ind', 'i4'),
            ('freq', 'f8'),
            ('intens', 'f8')
        ]

        # filter absorption intensities
        filt_intens = {filt.__name__ : [] for filt in self.filters}
        sizes = {
            filt.__name__ : np.zeros(world_size, dtype='i4')
            for filt in self.filters
        }
        jj_list = list(f['spec'].keys())
        ind = rank
        while ind < len(jj_list):
            j_bra, j_ket = [float(j) for j in jj_list[ind][1: -1].split(', ')]

            # spectrum and assignments
            spec = f['spec'][jj_list[ind]][:]
            assign_bra = f['assign'][str(j_bra)][:]
            assign_ket = f['assign'][str(j_ket)][:]

            # threshold
            if self.abs_intens_thresh > 0:
                spec = spec[
                    np.argwhere(
                        spec['intens'] > self.abs_intens_thresh
                    ).flatten()
                ]

            # spectrum indices to filter out
            filt_inds = np.empty(
                spec.size,
                dtype = [(filt.__name__, '?') for filt in self.filters]
            )
            for ind_ in range(spec.size):
                kwargs = {
                    'sym' : ( assign_bra['sym'][spec['bra ind'][ind_]],
                              assign_ket['sym'][spec['ket ind'][ind_]] ),
                    'qstr' : ( assign_bra['qstr'][spec['bra ind'][ind_]],
                               assign_ket['qstr'][spec['ket ind'][ind_]] )
                }
                for filt in self.filters:
                    filt_inds[filt.__name__][ind_] = filt(**kwargs)

            # apply filters
            for filt in self.filters:
                filt_spec = spec[filt_inds[filt.__name__]]

                # filter out intensities with indices
                filt_intens_ = np.empty(filt_spec.shape, dtype=intens_dtype)
                filt_intens_['J ket'] = np.ones(
                    filt_spec.shape, dtype='f4'
                ) * float(j_ket)
                filt_intens_['J bra'] = np.ones(
                    filt_spec.shape, dtype='f4'
                ) * float(j_bra)
                filt_intens_['ket ind'] = filt_spec['ket ind']
                filt_intens_['bra ind'] = filt_spec['bra ind']
                filt_intens_['freq'] = filt_spec['freq']
                filt_intens_['intens'] = filt_spec['intens']

                # save filtered intensities into dictionary
                filt_intens[filt.__name__].append(filt_intens_)
                sizes[filt.__name__][rank] += filt_inds[filt.__name__].size

            ind += world_size

        for filt in self.filters:

            # broadcast shapes...
            sizes_ = np.zeros(world_size, dtype='i4')
            comm.Allreduce(sizes[filt.__name__], sizes_, MPI.SUM)

            # ...to collectively create datasets...
            f.create_dataset(
                filt.__name__, (np.sum(sizes_), ), dtype=intens_dtype
            )

            # ...and write into datasets in parallel
            if not sizes_[rank] == 0:
                ind = np.sum(sizes_[: rank])
                for intens in filt_intens[filt.__name__]:
                    f[filt.__name__][ind: ind + intens.size] = intens
                    ind += intens.size

        f.close()

        if rank == 0:
            print('\n  FILTERED ABSORPTION INTENSITIES ...\n')
            for attr in ['filters', 'abs_intens_thresh']:
                print('      {} = {}'.format(attr, getattr(self, attr)))

        comm.Barrier()


    def _plot(self):
        """ Plots filtered absorption intensities. """

        # input / output file
        f = h5py.File(self.out_f, 'r', driver='mpio', comm=MPI.COMM_WORLD)
   
        if MPI.COMM_WORLD.Get_rank() == 0:

            colors = ['b', 'r', 'g']
            for ind, filt in enumerate(self.filters):
                ind_ = ind % len(colors)

                # frequencies and intensities
                spec = f[filt.__name__][:]

                # plot intensities
                plt.scatter(
                    spec['freq'],
                    spec['intens'],
                    s = 1,
                    c = colors[ind_],
                    label = '{} transitions'.format(filt.__name__)
                )

            # set plot parameters and save
            plt.xlabel(r'frequency ($cm^{-1}$)')
            plt.ylabel(r'intensity ($cm^{-1}$/($molecule \cdot cm^{-2}$))')
            plt.yscale('log')
            plt.ylim(bottom = self.abs_intens_thresh)
            plt.xlim(left = 0)
            plt.legend(loc='best')
            plt.title('absorption intensities')
            plt.tight_layout()
            plt.savefig('absorption_intensities.png', dpi=500, format='png')
            plt.close()

        # print to console
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('\n  PLOTTED ABSORPTION INTENSITIES ...\n')
            for attr in ['filters', 'abs_intens_thresh']:
                print('      {} = {}'.format(attr, getattr(self, attr)))
            print('')

        f.close()


    def totxt(self, form='default'):
        """ Writes assignments and spectrum into ASCII files.

            Args:
                form : str
                    The file format of the ASCII file. Set to 'default'.
        """

        # set txt form
        assert (form in ['default']), \
            f"form is unknown: '{form}' \
                (must be 'default')"

        # input / output file
        f = h5py.File(self.out_f, 'r', driver='mpio', comm=MPI.COMM_WORLD)

        if MPI.COMM_WORLD.Get_rank() == 0:

            if form == 'default':

                # write assignments TXT
                with open('assign.txt', 'w') as txt:
                    for j in list(f['assign'].keys()):
                        txt.write('\n    J = {}\n\n'.format(j))
                        np.savetxt(
                            txt, f['assign'][j][:], fmt='%10.4f %3s  %3.1f %s'
                        )

                # write spectrum TXT
                with open('spec.txt', 'w') as txt:
                    for jj in list(f['spec'].keys()):
                        txt.write('\n    (J_bra, J_ket) = {}\n\n'.format(jj))
                        np.savetxt(
                            txt,
                            f['spec'][jj][:],
                            fmt = '%6.1f %6.1f %12.6f %16.6e %16.6e'
                        )

            # TODO: implement other formats (e.g. HITRAN)

        # print to console
        if MPI.COMM_WORLD.Get_rank() == 0:
            print('  WRITTEN RESULTS INTO ASCII FILE ...\n')
            print('      format = {}\n'.format(form))

        f.close()


if __name__ == "__main__":


    # INITIALIZATION
    filename = 'matelem/hyfor_energies_f0.0_f39.0.chk'
    matelem = 'matelem/matelem_MU_ns_f<f1>_f<f2>.rchm'
    spec = field_free_spectrum(filename, matelem=matelem, j_max=2, e_max=15e3)

    # LINESTRENGTH
    spec.linestr(thresh=0)

    # ABSORPTION INTENSITY
    temp, part_sum= 296.0, 174.5813
    def allowed(**kwargs):
        pass_ns = True
        if 'qstr' in kwargs:
            qstr_bra, qstr_ket = kwargs['qstr']
            ns_bra = float(qstr_bra.split()[-1])
            ns_ket = float(qstr_ket.split()[-1])
            pass_ns = (ns_bra, ns_ket) in [(0.0, 0.0), (1.0, 1.0)]
        return pass_ns
    def forbidden(**kwargs):
        pass_ns = True
        if 'qstr' in kwargs:
            qstr_bra, qstr_ket = kwargs['qstr']
            ns_bra = float(qstr_bra.split()[-1])
            ns_ket = float(qstr_ket.split()[-1])
            pass_ns = (ns_bra, ns_ket) in [(0.0, 1.0), (1.0, 0.0)]
        return pass_ns
    spec.abs_intens(
        temp, part_sum, filters=[allowed, forbidden] , thresh=1e-36
    )

    # ASCII
    #spec.totxt(form='default')
