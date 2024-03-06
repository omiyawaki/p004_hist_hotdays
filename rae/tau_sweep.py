import os
import sys
from rae import rae
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

plev = np.linspace(1e5,1e3,50)
ps = 1000e2
tau0 = 3
Fs = 30
Fa = 150
b = 1
beta = 0
n = 2

odir='/project/amp/miyawaki/plots/p004/rae/tau_sweep'
if not os.path.exists(odir): os.makedirs(odir)

tau0s = np.linspace(1.5,10,50)

invstr = np.empty_like(tau0s)
OLR = np.empty_like(tau0s)
F1 = np.empty([len(tau0s), len(plev)])

for i in tqdm(range(len(tau0s))):
    _, _, invstr[i], OLR[i], F1[i,:] = rae(tau0s[i], b, beta, Fs, Fa, plev, ps, n)

fig, ax = plt.subplots()
ax.axhline(0, color='k', linewidth=0.5)
ax.plot(tau0s, invstr)
ax.set_xlabel('$\\tau_0$ (unitless)')
ax.set_ylabel('$T(\\tau_0) - T_s$ (K)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/inv_t0.png'%odir, format='png', dpi=600)

fig, ax = plt.subplots()
ax.axhline(0, color='k', linewidth=0.5)
ax.plot(tau0s, OLR)
ax.set_xlabel('$\\tau_0$ (unitless)')
ax.set_ylabel('OLR (Wm$^{-2}$)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/olr_t0.png'%odir, format='png', dpi=600)

