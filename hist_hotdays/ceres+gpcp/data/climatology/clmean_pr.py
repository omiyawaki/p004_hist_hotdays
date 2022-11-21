import os,sys
import pickle
import numpy as np
import xarray as xr

varn='pr'
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
    fn='%s/pr.gpcpd.200003-202202.nc'%idir
    ds=xr.open_dataset(fn)
    pr=ds[varn]
    # save grid info
    gr={}
    gr['lon']=ds['lon']
    gr['lat']=ds['lat']

    # select seasonal data if applicable
    if se!='ann':
        pr=pr.sel(time=pr['time.season']==se.upper())

    # take mean over time
    # mpr=pr.mean(dim='time',skipna=True).data
    pr=pr.data
    pr[~np.isfinite(pr)]=np.nan
    mpr=np.nanmean(pr, axis=0)

    pickle.dump([mpr, gr], open('%s/clmean.%s.%g-%g.%s.pickle' % (odir,varn,y0,y1,se), 'wb'), protocol=5)	
