import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
xpc=95
lse = ['djf'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 1970 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/era5/%s/%s' % (se,varn)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [dwt2m, gr] = pickle.load(open('%s/dwidth_%s.%s.%02d.%s.pickle' % (odir,yr,varn,xpc,se), 'rb'))

        # store data
        if c0 == 0:
            ydwt2m = np.empty([len(lyr),len(gr['lat']),len(gr['lon'])])
            c0 = 1

        ydwt2m[iyr,...] = dwt2m

    # take mean over time
    mt2m=np.mean(ydwt2m[:,...],axis=0)

    pickle.dump([mt2m, gr], open('%s/clmean.dw%s.%02d.%g.%g.%s.pickle' % (odir,varn,xpc,y0,y1,se), 'wb'), protocol=5)	
