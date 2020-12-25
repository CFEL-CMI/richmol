import numpy as np
import sys
import matplotlib.pyplot as plt

# compare transitions, plot stick spectrum

fname1 = sys.argv[1] # stark_transitions*.txt
fname2 = '../richmol2/MU_me'

freq1 = []
ls1 = []
with open(fname1,'r') as f:
    lines = f.readlines()
    for line in lines:
        w = line.split()
        freq1.append(float(w[24]))
        ls1.append(float(w[25]))

freq2 = []
ls2 = []
with open(fname2,'r') as f:
    lines = f.readlines()
    for line in lines:
        w = line.split()
        freq2.append(float(w[0]))
        ls2.append(-float(w[1]))

plt.plot(freq1, ls1, 'o')
plt.plot(freq2, ls2, 'o')
plt.show()
