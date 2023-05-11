import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='m2m'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2021 # end analysis year

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]

lpc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # available percentiles

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [hm2m, gr] = pickle.load(open('%s/hm2m_%s.%s.pickle' % (odir,yr,se), 'rb'))

        # store data
        if c0 == 0:
            yhm2m = np.empty([len(lyr),len(lpc),len(gr['lat']),len(gr['lon'])])
            c0 = 1

        yhm2m[iyr,...] = hm2m

    # take mean over time
    for ipc in range(len(lpc)):
        mm2m=np.mean(yhm2m[:,ipc,...],axis=0)

        pickle.dump([mm2m, gr], open('%s/clmean_m2m_%02d.%s.pickle' % (odir,lpc[ipc],se), 'wb'), protocol=5)	
