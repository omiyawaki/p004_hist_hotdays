import os,sys
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
import constants as c
from metpy.units import units

varn='ai1'
# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['ann','jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
y0=2000 # begin analysis year
y1=2022 # end analysis year

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]

for se in lse:
    idir='/project/amp/miyawaki/data/share/ai1/ceres+gpcp'
    odir='/project/amp/miyawaki/data/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # load data
    fn='%s/rn.ceres.200003-202202.nc'%idir
    ds=xr.open_dataset(fn)
    rn=ds['rn'].load()
    fn='%s/pr.gpcpd.200003-202202.nc'%idir
    ds=xr.open_dataset(fn)
    pr=ds['pr'].load()
    # save grid info
    gr={}
    gr['lon']=ds['lon']
    gr['lat']=ds['lat']

    # select seasonal data if applicable
    if se!='ann':
        rn=rn.sel(time=rn['time.season']==se.upper())
        pr=pr.sel(time=pr['time.season']==se.upper())

    # take means over time
    rn=rn.data
    rn[~np.isfinite(rn)]=np.nan
    rn=np.nanmean(rn, axis=0)
    pr=pr.data
    pr[~np.isfinite(pr)]=np.nan
    pr=np.nanmean(pr, axis=0)

    # compute ai
    rn=rn*units.joule/units.m**2/units.sec
    pr=pr*units.kg/units.m**2/units.sec
    ai1=rn/(c.Lv*pr)

    pickle.dump([ai1, gr], open('%s/clmean.mai1.%g-%g.%s.pickle' % (odir,y0,y1,se), 'wb'), protocol=5)	
