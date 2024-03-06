import sys
sys.path.append('/home/miyawaki/scripts/common')
from climlab.utils.thermo import rho_moist, pseudoadiabat
import constants as c
from akmaev_adjustment import convective_adjustment_direct

def rcae(tau0, b, beta, Fs, Fa, LH, SH, plev, ps, n):
    
    tau = tau0*(plev/ps)**n
    
    # calculate RAE profile first
    Ts = ( ( ( Fs*(2+tau0) + Fa*(1+b*tau0/(b+1)) - LH - SH) / (2+beta*tau0) ) / c.si )**(1/4)
    T = ( ( ((Fs+SH+Fa-beta*c.si*Ts**4)*(1+tau)+Fa*b/tau0*(tau/tau0)**(b-1) - Fa*tau0/(b+1)*(tau/tau0)**(b+1)) / (2*(1-beta)) ) / c.si )**(1/4)

    # convective adjustment
    # compute malr
    dTdp = pseudoadiabat(T,plev/100) / 100.  # K / Pa
    rho = plev/c.Rd/T  # in kg/m**3
    lapse=dTdp * c.g * rho * 1000.  # K / km
    print(lapse)
    
    return T, Ts
