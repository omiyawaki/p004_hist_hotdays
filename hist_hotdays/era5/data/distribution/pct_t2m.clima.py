import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

varn='t2m'
lse = ['djf'] # season (ann, djf, mam, jja, son)

# byr=[1959,2020] # year bounds
# byr=[1980,2000] # year bounds
byr=[2000,2022] # year bounds

pc = [1,5,50,95,99] # percentiles to compute

for se in lse:
    idir = '/project/mojave/observations/ERA5_daily/T2m'
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    fn = '%s/t2m_*.nc' % (idir)
    ds = xr.open_mfdataset(fn)
    # ds = xr.open_mfdataset(['%s/t2m_1978.nc'%idir,'%s/t2m_1979.nc'%idir])
    t2m = ds[varn]
    # select data within time of interest
    t2m=t2m.sel(time=t2m['time.year']>=byr[0])
    t2m=t2m.sel(time=t2m['time.year']<=byr[1])
    # select season
    if se != 'ann':
        t2m=t2m.sel(time=t2m['time.season']==se.upper())
    gr = {}
    gr['lon'] = ds['lon']
    gr['lat'] = ds['lat']
    t2m.load()

    # initialize array to store data
    ht2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

    # loop through gridir points to compute percentiles
    for ln in tqdm(range(gr['lon'].size)):
        for la in range(gr['lat'].size):
            lt = t2m[:,la,ln]
            nn=lt[~np.isnan(lt)] # remove nans
            ht2m[:,la,ln]=np.percentile(nn,pc)	

    pickle.dump([ht2m, gr], open('%s/h%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'wb'), protocol=5)	
