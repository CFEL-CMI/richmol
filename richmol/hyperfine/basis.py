import numpy as np
import itertools
from collections import deque


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
            List of spin quanta in coupled spin basis,
            e.g., [(I_1, I_12, ..., I_1N), (I_1, I_12, ..., I_1N), ...],
            where I_1, I_12, I_1N are quantum numbers of spin operators
            I_1, I_1+I_2, I_1+I_2+..I_N, respectively
        jQuanta : list
            List of rotational J quanta in coupled basis set
    """
    spinQueue = deque()
    spinQueue.append([spins[0]])
    for I in spins[1:]:
        nelem = len(spinQueue)
        for i in range(nelem):
            I0 = spinQueue.popleft()
            q = [I0 + [elem] for elem in np.arange(np.abs(I0[-1] - I), I0[-1] + I + 1)]
            spinQueue += q
    spinQueue = list(spinQueue)

    j = list(set([elem for spin in spinQueue for elem in np.arange(abs(f - spin[-1]), f + spin[-1] + 1)]))
    if not all(float(elem).is_integer() for elem in j):
        raise ValueError(f"Input values of spin quanta = {spins} and total F quantum number = {f} " + \
                         f"demand non-integer values of rotational J quanta = {j}") from None

    quanta = [s + [jj] for (s, jj) in itertools.product(spinQueue, j)
              if any(np.arange(np.abs(s[-1] - jj), s[-1] + jj + 1) == f)]

    spinQuanta = [elem[:-1] for elem in quanta]
    jQuanta = [int(elem[-1]) for elem in quanta]
    return spinQuanta, jQuanta



if __name__ == '__main__':

    spinQuanta, jQuanta = nearEqualCoupling(30, [1/2, 1/2])
    print(spinQuanta)
    print(jQuanta)
