import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

varn='t2m'
xpc=95 # percentile to take the distance from
lse = ['djf'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2020 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

lpc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # available percentiles
ixpc=np.where(np.equal(lpc,xpc))[0][0]
imed=np.where(np.equal(lpc,50))[0][0]

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/era5/%s/%s' % (se,varn)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [ht2m, gr] = pickle.load(open('%s/ht2m_%s.%s.pickle' % (odir,yr,se), 'rb'))

        dw=np.abs(ht2m[ixpc,...]-ht2m[imed,...])

        pickle.dump([dw,gr], open('%s/dwidth_%s.%s.%02d.%s.pickle' % (odir,yr,varn,xpc,se), 'wb'), protocol=5)	
