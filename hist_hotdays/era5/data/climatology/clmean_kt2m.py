import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
lre=['sea'] # can be empty
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
y0 = 2000 # begin analysis year
y1 = 2020 # end analysis year
bm=np.arange(285,310,0.1) # bins for computing pdf with kde

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

for re in lre:
    for se in lse:
        odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

        # load data
        c0=0 # first loop counter
        for iyr in tqdm(range(len(lyr))):
            yr = lyr[iyr]
            kt2m = pickle.load(open('%s/kt2m_%s.%s.%s.pickle' % (odir,yr,re,se), 'rb'))
            pde=kt2m(bm)

            # store data
            if c0 == 0:
                ykt2m = np.empty([len(lyr),len(bm)])
                c0 = 1

            ykt2m[iyr,:] = pde

        # take mean over time
        mt2m=np.mean(ykt2m,axis=0)

        pickle.dump([mt2m, bm], open('%s/clmean.kt2m.%s.%g.%g.%s.pickle' % (odir,re,y0,y1,se), 'wb'), protocol=5)	
