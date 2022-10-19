import os
import sys
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm

# this script aggregates the histogram of daily temperature for a given region on interest

varn='t2m'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

y0=1950 # first year
y1=1951 # last year+1

bn=np.arange(200.5,350.5,1) # bins

# file where selected region is provided
[sea,mgr]=pickle.load(open('/project/amp/miyawaki/plots/p004/hist_hotdays/cmip6/jja/fut-his/ssp245/mmm/defsea.t2m.95.ssp245.jja.pickle','rb'))
rlat=mgr[0][:,0]
rlon=mgr[1][0,:]

lyr=[str(y) for y in np.arange(y0,y1)]

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for yr in lyr:
        [ft2m, gr]=pickle.load(open('%s/f%s_%s.%s.pickle' % (odir,varn,yr,se), 'rb'))
        if np.logical_or(np.not_equal(gr['lat'],rlat).any(),np.not_equal(gr['lon'],rlon).any()):
            error('Check that selected region is in same grid as data being applied to.')
        nb=ft2m.shape[0]
        freg=np.zeros(nb)
        for ib in range(ft2m.shape[0]):
            lf=ft2m[ib,...]
            freg[ib]=lf[sea].sum()

        pickle.dump([freg, bn], open('%s/f%s_%s.sea.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
