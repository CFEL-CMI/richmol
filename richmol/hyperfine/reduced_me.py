import numpy as np
import py3nj
import inspect


_singleSpinMe = dict()

def register_me(func):
    _singleSpinMe[func.__name__.strip("_")] = func
    return func


@register_me
def _spin(I_bra, I_ket, tol=1e-8, **kwargs):
    """Reduced matrix element of spin operator <I_bra || I^(1) || I_ket>"""

    if round(float(I_bra), 1) == round(float(I_ket), 1):
        I2 = int(float(I_bra) * 2)
        threej = py3nj.wigner3j(I2, 1*2, I2,  -I2, 0, I2) 
        if abs(threej) < tol and I2 < tol:
            return 0
        elif abs(threej) < tol and I2 > tol:
            raise ValueError(f"Illegal division I/threej: '{I_bra} / {threej}'") from None
        else:
            return I_bra / threej
    else:
        return 0


@register_me
def _qmom(I_bra, I_ket, tol=1e-8, **kwargs):
    """Reduced matrix element of nuclear quadrupole moment <I_bra || Q^(2) || I_ket>"""

    try:
        eQ = float(kwargs["eQ"])
    except KeyError:
        raise KeyError(f"Please provide additional keyword argument 'eQ' " + \
                       f"(value of nuclear quadrupole moment) to function 'spinMe'") from None

    if round(float(I_bra), 1) == round(float(I_ket), 1):
        I2 = int(float(I_bra) * 2)
        threej = py3nj.wigner3j(I2, 2*2, I2,  -I2, 0, I2) 
        if abs(threej) < tol and abs(eQ) < tol:
            return 0
        elif abs(threej) < tol and abs(eQ) > tol:
            assert (I2 > 1), f"Not sure, but for nuclei with spin = {I_bra} " + \
                f"the nuclear quadrupole moment must be zero, instead got {eQ}"
            raise ValueError(f"Illegal division eQ/threej: '{eQ} / {threej}'") from None
        else:
            return 0.5 * eQ / threej
    else:
        return 0


def _spinMe(I_bra, I_ket, ispin, no_spins, spins, rank, oper, **kwargs):
    """Reduced matrix element of a single-spin operator in coupled spin basis"""

    I1_bra = I_bra[no_spins-1]
    I1_ket = I_ket[no_spins-1]
    I2_bra = spins[no_spins]
    I2_ket = spins[no_spins]
    I12_bra = I_bra[no_spins]
    I12_ket = I_ket[no_spins]

    if ispin == no_spins:
        if no_spins == 0:
            coef1 = 1
        elif no_spins > 0 and I1_bra != I1_ket:
            return 0
        else:
            fac = I1_bra + I2_ket + I12_bra + rank
            assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
            coef1 = (-1)**fac \
                  * np.sqrt((2*I12_bra + 1) * (2*I12_ket + 1)) \
                  * py3nj.wigner6j(int(I2_bra*2), int(I12_bra*2), int(I1_bra*2),
                                   int(I12_ket*2), int(I2_ket*2), int(rank*2))
        if oper in _singleSpinMe:
            coef2 = _singleSpinMe[oper](spins[ispin], spins[ispin], **kwargs)
        else:
            raise KeyError(f"Function '{oper}' is not available")

    else:
        if I2_bra != I2_ket:
            return 0
        else:
            fac = I1_bra + I2_bra + I12_ket + rank
            assert (float(fac).is_integer()), f"Non-integer power in (-1)**g: '(-1)**{fac}'"
            coef1 = (-1)**fac \
                  * np.sqrt((2*I12_bra + 1) * (2*I12_ket + 1)) \
                  * py3nj.wigner6j(int(I1_bra*2), int(I12_bra*2), int(I2_bra*2),
                                   int(I12_ket*2), int(I1_ket*2), int(rank*2))
            coef2 = _spinMe(I_bra, I_ket, ispin, no_spins-1, spins, rank, oper, **kwargs)

    return coef1 * coef2


def spinMe(quanta, spins, rank, oper, **kwargs):
    """Reduced matrix elements of a single-spin operator O(I_i)
    in coupled spin basis

    Args:
        quanta : list
            List of spin quanta in coupled spin basis,
            e.g., [(I_1, I_12, ..., I_1N), (I_1, I_12, ..., I_1N), ...],
            where I_1, I_12, I_1N are quantum numbers of spin operators
            I_1, I_1+I_2, I_1+I_2+..I_N, respectively
        spins : list
            List of spin quanta, e.g., [I_1, I_2, ... I_N]
        rank : int
            Rank of operator O(I_i)
        oper: str
            Operator identifier. Use 'qmom' for quadrupole moment
            operator, 'spin' for spin operator
        kwargs : set of additional parameters
            'eQ' : float
                Value of nuclear quadrupole constant (if oper == 'qmom')

    Returns:
        me : array (len(spins), len(quanta), len(quanta))
            Reduced matrix elements
            me[i, k, l] = < k || O(I_i) || l >
    """
    me = np.zeros((len(quanta), len(quanta), len(spins)), dtype=np.float64)
    for i, q1 in enumerate(quanta):
        for j, q2 in enumerate(quanta):
            for ispin in range(len(spins)):
                n = len(spins) - 1
                me[ispin, i, j]  = _spinMe(q1, q2, ispin, n, spins, rank, oper, **kwargs)
    return me


def spinMe_IxI(quanta, spins, rank):
    """Reduced matrix element of tensor product [I_i^(1) x I_j^(1)]^(rank)
    in coupled spin basis

    Args:
        quanta : list
            List of spin quanta in coupled spin basis,
            e.g., [(I_1, I_12, ..., I_1N), (I_1, I_12, ..., I_1N), ...],
            where I_1, I_12, I_1N are quantum numbers of spin operators
            I_1, I_1+I_2, I_1+I_2+..I_N, respectively
        spins : list
            List of spin quanta, e.g., [I_1, I_2, ... I_N]
        rank : int
            Rank of [I^(1) x I^(1)] tensor product, can be 0, 1, or 2

    Returns:
        me : array (len(spins), len(spins), len(quanta), len(quanta))
            Reduced matrix elements
            me[i, j, k, l] = < k || [I_i^(1) x I_j^(1)]^(rank) || l >
    """
    assert (rank in (0, 1, 2)), f"Illegal value of tensor product rank: '{rank}'"
    rme = spinMe(quanta, spins, 1, 'spin')
    coef = np.zeros((len(quanta), len(quanta), len(quanta)), dtype=np.float64)
    for i, q1 in enumerate(quanta):
        I1 = q1[-1]
        for j, q2 in enumerate(quanta):
            I2 = q2[-1]
            fac = I2 + I2 + rank
            assert (float(fac).is_integer()), f"Non-integer power in (-1)**f: '(-1)**{fac}'"
            fac = (-1)**fac * np.sqrt(2*rank + 1)
            n = len(quanta)
            coef[i, j, :] = fac * py3nj.wigner6j([2]*n, [2]*n, [rank*2]*n, [int(I2*2)]*n,
                                                 [int(I1*2)]*n, [q[-1]*2 for q in quanta])
    me = np.einsum('ikl,jln,knl->ijkn', rme, rme, coef)
    return me



if __name__ == '__main__':

    spins = [1/2, 1/2]
    quanta = [(1/2, 0), (1/2, 1)]

    print("matrix elements < || I_i || >")
    me = spinMe(quanta, spins, 1, 'spin')
    for k, q1 in enumerate(quanta):
        for l, q2 in enumerate(quanta):
            for i in range(len(spins)):
                print(q1, q2, i, me[i, k, l])


    print("matrix elements < || [I_i^(1) x I_j^(1)]^(2) || >")
    me = spinMe_IxI(quanta, spins, 2)
    for k, q1 in enumerate(quanta):
        for l, q2 in enumerate(quanta):
            for i in range(len(spins)):
                for j in range(len(spins)):
                    print(q1, q2, i, j, me[i, j, k, l])

