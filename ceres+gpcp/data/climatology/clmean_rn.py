import os,sys
import pickle
import numpy as np
import xarray as xr

varn='rn'
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
    rn=ds[varn]
    # save grid info
    gr={}
    gr['lon']=ds['lon']
    gr['lat']=ds['lat']

    # select seasonal data if applicable
    if se!='ann':
        rn=rn.sel(time=rn['time.season']==se.upper())

    # take mean over time
    # mrn=rn.mean(dim='time',skipna=True).data
    rn=rn.data
    rn[~np.isfinite(rn)]=np.nan
    mrn=np.nanmean(rn, axis=0)

    pickle.dump([mrn, gr], open('%s/clmean.%s.%g-%g.%s.pickle' % (odir,varn,y0,y1,se), 'wb'), protocol=5)	
