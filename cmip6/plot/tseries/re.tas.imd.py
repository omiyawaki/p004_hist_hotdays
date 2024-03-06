import os,sys
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from utils import monname
from metpy.units import units

md='CESM2'

mn=7
q=1
relb='se'
varn='tas'
se='ts'
dpi=600

fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)

idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
if not os.path.exists(odir): os.makedirs(odir)

tas=xr.open_dataarray('%s/re.%s.%s.%s.nc'%(idir,varn,se,relb))
tas=tas.sel(time=tas['time.month']==mn)
m=tas.groupby('time.year').mean('time')
p=tas.groupby('time.year').quantile(q,'time')
x=1850+np.arange(len(m))

# convert to deg F
def convert(v):
    v=v.data*units.K
    return v.to(units.degF).m
m=convert(m)
p=convert(p)

fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
ax.plot(x,m,'k',label='Average day')
ax.plot(x,p,'maroon',label='Hottest day')
ax.set_xlabel('Year')
ax.set_ylabel('Fahrenheit')
ax.set_title('Southeast US %s Temperature'%(monname(mn-1)))
ax.legend(frameon=False,fontsize=8)
fig.savefig('%s/re.%s.%s.%s.png'%(odir,varn,se,relb),format='png',dpi=dpi)

fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
ax.plot(x,p-m,'k')
ax.set_xlabel('Year')
ax.set_ylabel('Fahrenheit')
ax.set_title('Southeast US %s Hottest$-$Average Temperature'%(monname(mn-1)))
ax.legend(frameon=False,fontsize=8)
fig.savefig('%s/d.re.%s.%s.%s.png'%(odir,varn,se,relb),format='png',dpi=dpi)
