import os,sys
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

# tau0s = [2]
tau0s = np.linspace(1.5,10,50)

odir='/project/amp/miyawaki/plots/p004/rae/var_plev'
if not os.path.exists(odir): os.makedirs(odir)

T = np.empty([len(tau0s), len(plev)])
Ts = np.empty_like(tau0s)
invstr = np.empty_like(tau0s)
OLR = np.empty_like(tau0s)
F1 = np.empty([len(tau0s), len(plev)])

for i in tqdm(range(len(tau0s))):
    T[i,:], Ts[i], invstr[i], OLR[i], F1[i,:] = rae(tau0s[i], b, beta, Fs, Fa, plev, ps, n)

fig, ax = plt.subplots()
for i in tqdm(range(len(tau0s))):
    ax.plot(np.append(Ts[i], T[i,:]), 1e-2*np.append(ps, plev))
    # ax.plot(Ts[i], ps*1e-2, '.')
ax.set_ylim(ax.get_ylim()[::-1]) # invert r1 axis
ax.set_xlabel('T (K)')
ax.set_ylabel('p (hPa)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/t_plev.png'%odir, format='png', dpi=600)

fig, ax = plt.subplots()
for i in tqdm(range(len(tau0s))):
    ax.plot(F1[i,:], 1e-2*plev)
    # ax.plot(Ts[i], ps*1e-2, '.')
ax.set_ylim(ax.get_ylim()[::-1]) # invert r1 axis
ax.set_xlabel('$F_1$ (W m$^{-2}$)')
ax.set_ylabel('p (hPa)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/f1_plev.png'%odir, format='png', dpi=600)
