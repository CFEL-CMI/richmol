import numpy as np
import sys
import matplotlib.pyplot as plt

# compare transitions, plot stick spectrum and simple Gaussian crossections

fname1 = sys.argv[1] # stark_transitions*.txt
fname2 = '../richmol2/mu_me_z'

npoints = 100000
width = 0.000001
lineshape = lambda freq, ls0, freq0: ls0 * np.exp(-1.0/width * (freq - freq0)**2)

freq1 = []
ls1 = []
with open(fname1,'r') as f:
    lines = f.readlines()
    for line in lines:
        w = line.split()
        try:
            freq1.append(float(w[18]))
            ls1.append(float(w[19]))
        except IndexError:
            freq1.append(float(w[16]))
            ls1.append(float(w[17]))

freq_min = min(freq1)
freq_max = max(freq1)
grid1 = np.linspace(freq_min, freq_max, npoints)
profile1 = np.zeros(len(grid1), dtype=np.float64)

for fr,ls in zip(freq1, ls1):
    profile1 += lineshape(grid1, ls, fr)

freq2 = []
ls2 = []
with open(fname2,'r') as f:
    lines = f.readlines()
    for line in lines:
        w = line.split()
        freq2.append(float(w[0]))
        ls2.append(-float(w[1]))

freq_min = min(freq2)
freq_max = max(freq2)
grid2 = np.linspace(freq_min, freq_max, npoints)
profile2 = np.zeros(len(grid2), dtype=np.float64)

for fr,ls in zip(freq2, ls2):
    profile2 += lineshape(grid2, ls, fr)

plt.plot(freq1, ls1, 'o')
plt.plot(freq2, ls2, 'o')
plt.plot(grid1, profile1)
plt.plot(grid2, profile2)
plt.show()
