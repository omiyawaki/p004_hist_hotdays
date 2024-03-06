import os,sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from utils import varnlb

vn1='hfls'
vn2b='ooplh'
vn='%s+%sx'%(vn1,vn2b)
ld=np.concatenate(([10],np.arange(20,100,20),np.arange(100,850,50)))
lvn2=['%s%g'%(vn2b,d) for d in ld]
lvnd=['%s%s+%s%s'%('d',vn1,'d',vn2) for vn2 in lvn2]
lvndpc=['%s%s+%s%s'%('dpc',vn1,'dpc',vn2) for vn2 in lvn2]
lvnddpc=['%s%s+%s%s'%('ddpc',vn1,'ddpc',vn2) for vn2 in lvn2]

md='CESM2'
pc=97.5
se='sc'
sc='jja'
re='tr'
fo1='historical' # forcings 
fo2='ssp370' # forcings 
fo='%s-%s'%(fo2,fo1)
his='1980-2000'
fut='gwl2.0'

def relb(re):
    d={
            'tr':   'Tropics'
            }
    return d[re]

def sclb(sc):
    d={
            'jja':   'JJA+DJF'
            }
    return d[sc]

# dirs
odir='/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn)
if not os.path.exists(odir): os.makedirs(odir)
idird=['/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn) for vn in lvnd]
idirdpc=['/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn) for vn in lvndpc]
idirddpc=['/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,md,vn) for vn in lvnddpc]

# load correlation coefs
rd=[xr.open_dataarray('%s/sp.rsq.%s_%s_%s.%s.%s.nc' % (idir,vn,his,fut,sc,re)).data.item() for idir,vn in zip(idird,lvnd)]
rdpc=[xr.open_dataarray('%s/sp.rsq.%s_%s_%s.%s.%s.nc' % (idir,vn,his,fut,sc,re)).sel(pct=pc).data.item() for idir,vn in zip(idirdpc,lvndpc)]
rddpc=[xr.open_dataarray('%s/sp.rsq.%s_%s_%s.%s.%s.nc' % (idir,vn,his,fut,sc,re)).sel(pct=pc).data.item() for idir,vn in zip(idirddpc,lvnddpc)]

# plot correlation as function of mrsox
fig,ax=plt.subplots(figsize=(4,3),constrained_layout=True)
ax.plot(ld,rd,'.-k',label='Mean')
ax.plot(ld,rdpc,'.-',color='tab:red',label='Hot')
ax.plot(ld,rddpc,'.-',color='tab:orange',label='Hot$-$Mean')
ax.set_xlabel('Soil moisture depth (cm)')
ax.set_ylabel('$R({%s},{%s})$ (unitless)'%(varnlb(vn1),varnlb(vn2b)))
ax.set_title('%s %s %s'%(md,relb(re),sclb(sc)))
ax.set_ylim([0,1])
ax.legend(frameon=False)
fig.savefig('%s/sp.r.%s.%s.%s.%s.png' % (odir,vn,fo,sc,re), format='png', dpi=600)
fig.savefig('%s/sp.r.%s.%s.%s.%s.pdf' % (odir,vn,fo,sc,re), format='pdf', dpi=600)
