import os,sys
from rae import rae
from rae_tf import rae_tf
from rae_arb import rae_arb
# from rcae import rcae
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

plev = np.linspace(1e5,1e3,50)
ps = 1000e2
LH = 0
SH = -15
Fs = 0
Fa = 150
b = 1
beta = 0.2
n = 2
# Qa=n*b*Fa/ps*(plev/ps)**(n*b-1)
Qa=np.arange(0.029,0,-0.029/len(plev))/9.81
fbl=0.8 # fractional boundary layer depth

tau0s = [3]
taubl = [fbl*tau0 for tau0 in tau0s]
# tau0s = np.linspace(1.5,10,50)

odir='/project/amp/miyawaki/plots/p004/rae/comp'
if not os.path.exists(odir): os.makedirs(odir)

# caT = np.empty([len(tau0s), len(plev)])
# caTs = np.empty_like(tau0s)

# for i in tqdm(range(len(tau0s))):
#     caT[i,:], caTs[i] = rcae(tau0s[i], b, beta, Fs, Fa, LH, SH, plev, ps, n)

# RAE
T = np.empty([len(tau0s), len(plev)])
Ts = np.empty_like(tau0s)
for i in tqdm(range(len(tau0s))):
    T[i,:], Ts[i], _,_,_ = rae(tau0s[i], b, beta, Fs, Fa, plev, ps, n)

# RAE + turbulent flux
tfT = np.empty([len(tau0s), len(plev)])
tfTs = np.empty_like(tau0s)
for i in tqdm(range(len(tau0s))):
    tfT[i,:], tfTs[i], _,_,_ = rae_tf(tau0s[i], taubl[i], b, beta, Fs, Fa, LH, SH, plev, ps, n)

# RAE + turbulent flux + arbitrary heating
arbT = np.empty([len(tau0s), len(plev)])
arbTs = np.empty_like(tau0s)
for i in tqdm(range(len(tau0s))):
    arbT[i,:], arbTs[i], _,_,_ = rae_arb(tau0s[i], taubl[i], b, beta, Fs, Qa, LH, SH, plev, ps, n)

fig, ax = plt.subplots()
for i in tqdm(range(len(tau0s))):
    ax.plot(np.append(Ts[i], T[i,:]), 1e-2*np.append(ps, plev), color='tab:blue')
    ax.plot(np.append(tfTs[i], tfT[i,:]), 1e-2*np.append(ps, plev), '--',color='tab:orange')
    ax.plot(np.append(arbTs[i], arbT[i,:]), 1e-2*np.append(ps, plev), '-',color='tab:red')
    # ax.plot(Ts[i], ps*1e-2, '.')
ax.set_ylim(ax.get_ylim()[::-1]) # invert r1 axis
ax.set_xlabel('T (K)')
ax.set_ylabel('p (hPa)')
fig.set_size_inches(4,3)
plt.tight_layout()
plt.savefig('%s/t_plev.png'%odir, format='png', dpi=600)

