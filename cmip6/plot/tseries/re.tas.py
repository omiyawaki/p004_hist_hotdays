import os,sys
sys.path.append('../../data')
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from tqdm import tqdm
from util import mods
from utils import monname
from metpy.units import units
from scipy.ndimage import uniform_filter1d as uf1d
from regions import retname

md='CESM2'

mn=7
q=1
nf=20 # years for rolling mean
relb='us'
renm=retname(relb)
varn='tas'
se='ts'
dpi=600

fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)

md='mmm'
lmd=mods(fo1)

odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)

def load_vn(md):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    if not os.path.exists(odir): os.makedirs(odir)

    tas=xr.open_dataarray('%s/re.%s.%s.%s.nc'%(idir,varn,se,relb))
    tas=tas.sel(time=tas['time.month']==mn)
    m=tas.groupby('time.year').mean('time')
    p=tas.groupby('time.year').quantile(q,'time')
    return m,p

for i,md in enumerate(tqdm(lmd)):
    if i==0:
        m,p=load_vn(md)
    else:
        m0,p0=load_vn(md)
        m=xr.concat([m,m0],'model')
        p=xr.concat([p,p0],'model')

ms=m.std('model')
ps=p.std('model')
m=m.mean('model')
p=p.mean('model')
x=1850+np.arange(len(m))

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,m-273.15,'k',label='Average day')
ax.set_xlabel('Year')
ax.set_ylabel('Celsius')
ax.set_title('%s %s Temperature'%(renm,monname(mn-1)))
ax.legend(frameon=False,fontsize=10)
fig.savefig('%s/re.%s.%s.%s.degc.avgonly.png'%(odir,varn,se,relb),format='png',dpi=dpi)

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,m-273.15,'k',label='Average day')
ax.plot(x,p-273.15,'maroon',label='Hottest day')
ax.set_xlabel('Year')
ax.set_ylabel('Celsius')
ax.set_title('%s %s Temperature'%(renm,monname(mn-1)))
ax.legend(frameon=False,fontsize=10)
fig.savefig('%s/re.%s.%s.%s.degc.png'%(odir,varn,se,relb),format='png',dpi=dpi)

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,uf1d(p-m,nf),'k')
ax.set_xlabel('Year')
ax.set_ylabel('Celsius')
ax.set_title('Hottest$-$Average %s Temperature'%(monname(mn-1)))
fig.savefig('%s/d.re.%s.%s.%s.degc.png'%(odir,varn,se,relb),format='png',dpi=dpi)

# convert to deg F
def convert(v):
    v=v.data*units.K
    return v.to(units.degF).m
ms=convert(ms)
ps=convert(ps)
m=convert(m)
p=convert(p)

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,m,'k',label='Average day')
ax.plot(x,p,'maroon',label='Hottest day')
ax.set_xlabel('Year')
ax.set_ylabel('Fahrenheit')
ax.set_title('%s %s Temperature'%(renm,monname(mn-1)))
ax.legend(frameon=False,fontsize=10)
fig.savefig('%s/re.%s.%s.%s.png'%(odir,varn,se,relb),format='png',dpi=dpi)

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,uf1d(p-m,nf),'k')
ax.set_xlabel('Year')
ax.set_ylabel('Fahrenheit')
ax.set_title('Hottest$-$Average %s Temperature'%(monname(mn-1)))
fig.savefig('%s/d.re.%s.%s.%s.png'%(odir,varn,se,relb),format='png',dpi=dpi)
