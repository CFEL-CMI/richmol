import numpy as np
import itertools
from collections import deque


def spinNearEqualCoupling(spins):
    """Generates combinations of nuclear spin quanta
    following nearly-equal coupling scheme

    Args:
        spins : list
            List of nuclear spin quanta

    Returns:
        spinQuanta : list
            List of spin quanta in coupled basis,
            e.g., [(I_1, I_12, ..., I_1N), (I_1, I_12, ..., I_1N), ...],
            where I_1, I_12, I_1N are quantum numbers of spin operators
            I_1, I_1+I_2, I_1+I_2+..I_N, respectively
    """
    spinQueue = deque()
    spinQueue.append([spins[0]])
    for I in spins[1:]:
        nelem = len(spinQueue)
        for i in range(nelem):
            I0 = spinQueue.popleft()
            q = [I0 + [elem] for elem in np.arange(np.abs(I0[-1] - I), I0[-1] + I + 1)]
            spinQueue += q
    spinQuanta = [tuple(elem) for elem in spinQueue]
    return spinQuanta


def nearEqualCoupling(f, spins):
    """Generates combinations of nuclear spin and rotational J quanta
    following nearly-equal coupling scheme

    Args:
        f : float
            Value of quantum number F of the total angular momentum
            F = I_1 + I_2 +... I_N + J
        spins : list
            List of nuclear spin quanta

    Returns:
        spinQuanta : list
            List of spin quanta in coupled basis,
            e.g., [(I_1, I_12, ..., I_1N), (I_1, I_12, ..., I_1N), ...],
            where I_1, I_12, I_1N are quantum numbers of spin operators
            I_1, I_1+I_2, I_1+I_2+..I_N, respectively
        jQuanta : list
            List of rotational J quanta in coupled basis set
    """
    spinBasis = spinNearEqualCoupling(spins)

    j = list(set([elem for spin in spinBasis for elem in np.arange(abs(f - spin[-1]), f + spin[-1] + 1)]))
    if not all(float(elem).is_integer() for elem in j):
        raise ValueError(f"Input values of spin quanta = {spins} and total F quantum number = {f} " + \
                         f"demand non-integer values of rotational J quanta = {j}") from None

    quanta = [(s, jj) for (s, jj) in itertools.product(spinBasis, j)
              if any(np.arange(np.abs(s[-1] - jj), s[-1] + jj + 1) == f)]

    spinQuanta = [elem[0] for elem in quanta]
    jQuanta = [int(elem[1]) for elem in quanta]
    return spinQuanta, jQuanta



if __name__ == '__main__':

    # test for F = 30 and two spins 1/2

    spinQuanta, jQuanta = nearEqualCoupling(30, [1/2, 1/2])
    print(spinQuanta)
    print(jQuanta)
