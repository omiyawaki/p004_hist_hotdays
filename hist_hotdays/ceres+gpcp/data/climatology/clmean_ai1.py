import os,sys
import pickle
import numpy as np
import xarray as xr

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
    fn='%s/ai1.ceres.gpcp.200003-202202.nc'%idir
    ds=xr.open_dataset(fn)
    ai1=ds[varn]
    # save grid info
    gr={}
    gr['lon']=ds['lon']
    gr['lat']=ds['lat']

    # select seasonal data if applicable
    if se!='ann':
        ai1=ai1.sel(time=ai1['time.season']==se.upper())

    # take mean over time
    # mai1=ai1.mean(dim='time',skipna=True).data
    ai1=ai1.data
    ai1[~np.isfinite(ai1)]=np.nan
    mai1=np.nanmean(ai1, axis=0)

    pickle.dump([mai1, gr], open('%s/clmean.%s.%g-%g.%s.pickle' % (odir,varn,y0,y1,se), 'wb'), protocol=5)	
