import sys
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import constants as c
from scipy.integrate import trapezoid as trapz
from scipy.integrate import cumulative_trapezoid as cumtrapz

def rae_arb(tau0, taubl, b, beta, Fs, Qa, LH, SH, plev, ps, n):
    
    tau = tau0*(plev/ps)**n

    # boundary layer (BL) and free atmosphere (FA) masks
    hbl=np.ones_like(tau)
    hbl[tau<taubl]=0
    hfa=np.zeros_like(tau)
    hfa[tau<taubl]=1

    # atm heating integrals
    Fa=-np.trapz(Qa,plev) # vertically integrated heating
    cnv=n*tau0/ps*(plev/ps)**(n-1) # conversion factor to ensure vertical integral in tau = p
    Fat=-np.trapz(Qa/cnv,tau) # vertically integrated heating
    g1=-cumtrapz(Qa/cnv,tau,initial=0)
    g2=np.flip(cumtrapz(np.flip(g1),np.flip(tau),initial=0))
    print('Fa = %g W m**-2'%Fa)
    print(Fs*(2+tau0) + Fa + g2[0] - LH*(2+tau0) - SH*(1+(tau0-taubl)/3))
    
    # ramp
    Ts = ( ( ( Fs*(2+tau0) + Fa + g2[0] - LH*(2+tau0) - SH*(1+(tau0-taubl)/3) ) / (2+beta*tau0) ) / c.si )**(1/4)
    T = ( ( ( (Fs-LH-beta*c.si*Ts**4)*(1+tau)+ Fa+g2+Qa/cnv +SH/(tau0-taubl)**2*(2*(tau-taubl)-(tau-taubl)**3/3)*hbl ) / (2*(1-beta)) ) / c.si )**(1/4)

    OLR = ((Fs+SH+Fa-beta*c.si*Ts**4)*(1) ) / (2*(1-beta))

    F1 = Fs+SH+Fa*(1-(tau/tau0)**b)-beta*c.si*Ts**4
    
    if plev[1]-plev[0] > 0:
        Tlev0 = T[-1]
    else:
        Tlev0 = T[0]

    invstr = Tlev0 - Ts
    
    return T, Ts, invstr, OLR, F1

# # FOR TESTING:
# plev=np.linspace(1e5,1e3,50)
# ps=1000e2
# tau0=2
# taubl=tau0*0.9
# b=1
# beta=0.2
# Fs=0
# LH=0
# SH=-5
# n=2
# Fa=100
# Qa=c.g*n*b*Fa/ps*(plev/ps)**(n*b-1)
# rae_arb(tau0,taubl,b,beta,Fs,Qa,LH,SH,plev,ps,n)
