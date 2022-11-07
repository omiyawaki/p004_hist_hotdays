import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

varn='pr'
lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

y0=1997 # first year
y1=2021 # last year+1

lyr=[str(y) for y in np.arange(y0,y1)]

pc = [1,5,50,95,99] # percentiles to compute

for se in lse:
    idir = '/project/mojave/observations/OBS-PR/GPCP_DAILY'
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for yr in lyr:
        fn = '%s/gpcp.v01r03.daily.%s.nc' % (idir,yr)
        ds = xr.open_dataset(fn)
        pr = ds['precip']
        if se != 'ann':
            pr=pr.sel(time=pr['time.season']==se.upper())
        gr = {}
        gr['lon'] = ds['longitude']
        gr['lat'] = ds['latitude']

        # initialize array to store data
        hpr = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

        # loop through gridir points to compute percentiles
        for ln in tqdm(range(gr['lon'].size)):
            for la in range(gr['lat'].size):
                lt = pr[:,la,ln]
                hpr[:,la,ln]=np.percentile(lt,pc)	

        pickle.dump([hpr, gr], open('%s/h%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
