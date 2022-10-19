import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import gaussian_kde

# this script aggregates the histogram of daily temperature for a given region on interest

varn='t2m'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

y0=1950 # first year
y1=2021 # last year+1

bn=np.arange(200.5,350.5,1) # bins

# file where selected region is provided
[sea,mgr]=pickle.load(open('/project/amp/miyawaki/plots/p004/hist_hotdays/cmip6/jja/fut-his/ssp245/mmm/defsea.t2m.95.ssp245.jja.pickle','rb'))
rlat=mgr[0][:,0]
rlon=mgr[1][0,:]

lyr=[str(y) for y in np.arange(y0,y1)]

for se in lse:
    idir = '/project/mojave/observations/ERA5_daily/T2m'
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for yr in tqdm(lyr):
        fn = '%s/t2m_%s.nc' % (idir,yr)
        ds = xr.open_dataset(fn)
        t2m = ds['t2m'].load()
        if se != 'ann':
            t2m=t2m.sel(time=t2m['time.season']==se.upper())
        gr = {}
        gr['lon'] = ds['lon']
        gr['lat'] = ds['lat']

        if np.logical_or(np.not_equal(gr['lat'],rlat).any(),np.not_equal(gr['lon'],rlon).any()):
            error('Check that selected region is in same grid as data being applied to.')

        rt2m=np.empty([t2m.shape[0],len(sea[0])])
        for it in range(t2m.shape[0]):
            lt=t2m[it,...].data
            rt2m[it,:]=lt[sea] # regionally selected data
        rt2m=rt2m.flatten()
        kt2m=gaussian_kde(rt2m)

        pickle.dump(kt2m, open('%s/k%s_%s.sea.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
