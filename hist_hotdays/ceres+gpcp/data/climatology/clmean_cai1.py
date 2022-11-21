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
    fn='%s/cai1.ceres.gpcp.200003-202202.nc'%idir
    ds=xr.open_dataset(fn)
    ai1=ds[varn]
    ai1=ai1.data
    # save grid info
    gr={}
    gr['lon']=ds['lon']
    gr['lat']=ds['lat']

    # select seasonal data if applicable
    if se=='djf':
        ai1=np.roll(ai1,1,axis=0)
        ai1=ai1[0:3]
    elif se=='mam':
        ai1=ai1[2:5]
    elif se=='jja':
        ai1=ai1[5:8]
    elif se=='son':
        ai1=ai1[8:11]

    # take mean over time
    # mai1=ai1.mean(dim='time',skipna=True).data
    ai1[~np.isfinite(ai1)]=np.nan
    mai1=np.nanmean(ai1, axis=0)
    print(np.nanmax(mai1))
    sys.exit()

    pickle.dump([mai1, gr], open('%s/clmean.cai1.%g-%g.%s.pickle' % (odir,y0,y1,se), 'wb'), protocol=5)	
