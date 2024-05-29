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

# mn=7
mn1='summer'
mn2='winter'
q=1
nf=20 # years for rolling mean
relb='tr_lnd'
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

def monstr(m):
    if type(m) is int:
        return monname(m-1)
    else:
        return m.title()


def load_vn(md,mn):
    idir='/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s'%(se,fo,md,varn)
    if not os.path.exists(odir): os.makedirs(odir)

    if type(mn) is int:
        tas=xr.open_dataarray('%s/re.%s.%s.%s.nc'%(idir,varn,se,relb))
        tas=tas.sel(time=tas['time.month']==mn)
        m=tas.groupby('time.year').mean('time')
        p=tas.groupby('time.year').quantile(q,'time')
    else:
        ds=xr.open_dataset('%s/re.%s.%s.%s.%s.nc'%(idir,varn,se,relb,mn))
    return ds['mtas'],ds['ptas']

for i,md in enumerate(tqdm(lmd)):
    if i==0:
        ms,ps=load_vn(md,mn1)
        mw,pw=load_vn(md,mn2)
    else:
        ms0,ps0=load_vn(md,mn1)
        mw0,pw0=load_vn(md,mn2)
        ms=xr.concat([ms,ms0],'model')
        ps=xr.concat([ps,ps0],'model')
        mw=xr.concat([mw,mw0],'model')
        pw=xr.concat([pw,pw0],'model')

mss=ms.std('model')
pss=ps.std('model')
ms=ms.mean('model')
ps=ps.mean('model')
mws=mw.std('model')
pws=pw.std('model')
mw=mw.mean('model')
pw=pw.mean('model')
x=1850+np.arange(len(ms))

# climatology
cds=(ps-ms).sel(year=ms['year'].isin(np.arange(1850,1900,1))).mean('year')
cdw=(pw-mw).sel(year=ms['year'].isin(np.arange(1850,1900,1))).mean('year')

fig,ax=plt.subplots(figsize=(6,3),constrained_layout=True)
ax.plot(x,uf1d(ps-ms,nf)-cds.data,'tab:orange',label='Summer')
ax.plot(x,uf1d(pw-mw,nf)-cdw.data,'tab:blue',label='Winter')
ax.set_xlabel('Year')
ax.set_ylabel('Celsius')
ax.set_title('Hottest$-$Median Temperature Change')
ax.legend(frameon=False,fontsize=10)
fig.savefig('%s/d.re.%s.%s.%s.sumwin.degc.png'%(odir,varn,se,relb),format='png',dpi=dpi)
