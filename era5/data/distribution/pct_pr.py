import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

varn='pr'
lse = ['jja'] # season (ann, djf, mam, jja, son)

y0=1950 # first year
y1=2021 # last year+1

lyr=[str(y) for y in np.arange(y0,y1)]
lmn=['%02d' % m for m in np.arange(1,12+1)]

pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # percentiles to compute

for se in lse:
    idir = '/project/mojave/observations/ERA5_daily/PR'
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for yr in tqdm(lyr):
        for mn in lmn:
            fn = '%s/PR_%s%s.nc' % (idir,yr,mn)
            ds = xr.open_dataset(fn)
            if mn=='01':
                pr = ds['PR']
            else:
                pr=xr.concat((pr,ds['PR']),'time')

        if se != 'ann':
            pr=pr.sel(time=pr['time.season']==se.upper())
        gr = {}
        gr['lon'] = ds['lon']
        gr['lat'] = ds['lat']

        # initialize array to store data
        hpr = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

        # loop through gridir points to compute percentiles
        for ln in tqdm(range(gr['lon'].size)):
            for la in range(gr['lat'].size):
                lt = pr[:,la,ln]
                hpr[:,la,ln]=np.percentile(lt,pc)	

        pickle.dump([hpr, gr], open('%s/h%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
