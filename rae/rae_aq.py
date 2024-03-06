import sys
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import constants as c

def rae_aq(tau0, taubl, b, beta, Fs, Qa, LH, SH, plev, ps, n):
    # calculate rae with an arbitrary (prescribed) Qa profile
    
    tau = tau0*(plev/ps)**n

    # boundary layer (BL) mask
    hbl=np.ones_like(tau)
    hbl[tau<taubl]=0

    # ramp
    Ts = ( ( ( Fs*(2+tau0) + Fa*(1+b*tau0/(b+1)) - LH*(2+tau0) - SH*(1+(tau0-taubl)/3) ) / (2+beta*tau0) ) / c.si )**(1/4)
    T = ( ( ( (Fs-LH+Fa-beta*c.si*Ts**4)*(1+tau)+Fa*b/tau0*(tau/tau0)**(b-1) - Fa*tau0/(b+1)*(tau/tau0)**(b+1) +SH/(tau0-taubl)**2*(2*(tau-taubl)-(tau-taubl)**3/3)*hbl ) / (2*(1-beta)) ) / c.si )**(1/4)

    OLR = ((Fs+SH+Fa-beta*c.si*Ts**4)*(1) ) / (2*(1-beta))

    F1 = Fs+SH+Fa*(1-(tau/tau0)**b)-beta*c.si*Ts**4
    
    if plev[1]-plev[0] > 0:
        Tlev0 = T[-1]
    else:
        Tlev0 = T[0]

    invstr = Tlev0 - Ts
    
    return T, Ts, invstr, OLR, F1
