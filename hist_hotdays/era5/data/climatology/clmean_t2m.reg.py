import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
lre=['swus']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 1970 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

for re in lre:
    for se in lse:
        odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        # load data
        c0=0 # first loop counter
        for iyr in tqdm(range(len(lyr))):
            yr = lyr[iyr]
            [ht2m, lpc] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (odir,yr,re,se), 'rb'))

            # store data
            if c0 == 0:
                yht2m = np.empty([len(lyr),len(lpc)])
                c0 = 1

            yht2m[iyr,:] = ht2m

        # take mean over time
        mt2m=np.mean(yht2m,axis=0)

        pickle.dump([mt2m, lpc], open('%s/clmean.%s.%s.%g.%g.%s.pickle' % (odir,varn,re,y0,y1,se), 'wb'), protocol=5)	
