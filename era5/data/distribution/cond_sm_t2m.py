import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

varn='sm' # variable name

lse = ['djf'] # season (ann, djf, mam, jja, son)
# byr=[1959,2020] # output year bounds
byr=[1980,2000] # output year bounds

# percentiles
pc = [0,95,99] 

for se in lse:
    idirv='/project/mojave/observations/ERA5_daily/soilmoisture_10cm'
    idirt='/project/mojave/observations/ERA5_daily/T2m'
    odirt='/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,'t2m')
    odir='/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    c=0 # counter
    # load temp
    fn = '%s/%s_*.nc' % (idirt,'t2m')
    ds = xr.open_mfdataset(fn)
    # save grid info
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']
    t2m=ds['t2m'].load()
    del(ds)
    # load varn
    fn = '%s/%s_*.nc' % (idirv,'soilmoisture_10cm')
    ds = xr.open_mfdataset(fn)
    sm=ds['soilmoisture10'].load()
    del(ds)
        
    # select data within time of interest
    t2m=t2m.sel(time=t2m['time.year']>=byr[0])
    t2m=t2m.sel(time=t2m['time.year']<=byr[1])
    sm=sm.sel(time=sm['time.year']>=byr[0])
    sm=sm.sel(time=sm['time.year']<=byr[1])

    # select seasonal data if applicable
    if se != 'ann':
        t2m=t2m.sel(time=t2m['time.season']==se.upper())
        sm=sm.sel(time=sm['time.season']==se.upper())
    
    # load percentile values
    ht2m,_=pickle.load(open('%s/h%s_%g-%g.%s.pickle' % (odirt,'t2m',byr[0],byr[1],se), 'rb'))	
    ht2m95=ht2m[-2,...]
    ht2m99=ht2m[-1,...]

    # initialize array to store subsampled means data
    csm = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

    # loop through gridir points to compute percentiles
    for ln in tqdm(range(gr['lon'].size)):
        for la in range(gr['lat'].size):
            lt = t2m[:,la,ln]
            lv = sm[:,la,ln]
            lt95 = ht2m95[la,ln]
            lt99 = ht2m99[la,ln]
            csm[0,la,ln]=np.nanmean(lv)
            csm[1,la,ln]=np.nanmean(lv[np.where(lt>lt95)])
            csm[2,la,ln]=np.nanmean(lv[np.where(lt>lt99)])

    pickle.dump([csm, gr], open('%s/c%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'wb'), protocol=5)	
