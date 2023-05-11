import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='pr'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
y0 = 1997 # begin analysis year
y1 = 2020 # end analysis year

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]

lpc = [1,5,50,95,99] # available percentiles

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/gpcp/%s/%s' % (se,varn)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [hpr, gr] = pickle.load(open('%s/hpr_%s.%s.pickle' % (odir,yr,se), 'rb'))

        # store data
        if c0 == 0:
            yhpr = np.empty([len(lyr),len(lpc),len(gr['lat']),len(gr['lon'])])
            c0 = 1

        yhpr[iyr,...] = hpr

    # take mean over time
    for ipc in range(len(lpc)):
        mpr=np.nanmean(yhpr[:,ipc,...],axis=0)

        pickle.dump([mpr, gr], open('%s/clmean_%02d.%s.pickle' % (odir,lpc[ipc],se), 'wb'), protocol=5)	
