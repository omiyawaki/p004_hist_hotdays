import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
xpc='95'
lre=['sea_mp']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2020 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

# file where selected region is provided
[sea,mgr]=pickle.load(open('/project/amp/miyawaki/plots/p004/hist_hotdays/cmip6/jja/fut-his/ssp245/mmm/defsea.t2m.95.ssp245.jja.pickle','rb'))
rlat=mgr[0][:,0]
rlon=mgr[1][0,:]
# for this case just look at one point (median of selected region)
pla=int(np.median(sea[0]))
plo=int(np.median(sea[1]))

for re in lre:
    for se in lse:
        odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        # load data
        c0=0 # first loop counter
        for iyr in tqdm(range(len(lyr))):
            yr = lyr[iyr]
            [dwt2m, gr] = pickle.load(open('%s/dwidth_%s.%s.%s.%s.pickle' % (odir,yr,varn,xpc,se), 'rb'))

            # store data
            if c0 == 0:
                ydwt2m = np.empty(len(lyr))
                c0 = 1

            ydwt2m[iyr] = dwt2m[pla,plo]

        pickle.dump(ydwt2m, open('%s/tseries.dw%s.%s.%s.%g.%g.%s.pickle' % (odir,varn,xpc,re,y0,y1,se), 'wb'), protocol=5)	
