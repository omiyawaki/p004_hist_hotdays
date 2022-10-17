import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint as td2q
import constants as c

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

varn='m2m'
lse = ['djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

y0=1950 # first year
y1=2021 # last year+1

lyr=[str(y) for y in np.arange(y0,y1)]

pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # percentiles to compute

for se in lse:
    idir = '/project/mojave/observations/ERA5_daily'
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for yr in lyr:
        # load 2 m T
        fn = '%s/T2m/t2m_%s.nc' % (idir,yr)
        ds = xr.open_dataset(fn)
        t2m = ds['t2m']*units.kelvin
        # load 2 m dew point T
        fn = '%s/TD2m/td2m_%s.nc' % (idir,yr)
        ds = xr.open_dataset(fn)
        td2m = ds['td2m']*units.kelvin
        # load surface pressure
        fn = '%s/PS/ps_%s.nc' % (idir,yr)
        ds = xr.open_dataset(fn)
        ps = ds['ps']*units.pascal
        # calculate specific humidity
        q2m = td2q(ps,td2m)
        # compute MSE
        m2m=c.cpd*t2m+c.Lv*q2m

        # gc
        t2m.close()
        td2m.close()
        ps.close()

        # select by season if appropriate
        if se != 'ann':
            m2m=m2m.sel(time=m2m['time.season']==se.upper())
        gr = {}
        gr['lon'] = ds['lon']
        gr['lat'] = ds['lat']

        # initialize array to store data
        hm2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

        # loop through gridir points to compute percentiles
        for ln in tqdm(range(gr['lon'].size)):
            for la in range(gr['lat'].size):
                lt = m2m[:,la,ln]
                hm2m[:,la,ln]=np.percentile(lt,pc)	

        pickle.dump([hm2m, gr], open('%s/h%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
