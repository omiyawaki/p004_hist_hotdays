import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
varn='t2m'
xpc=95
lse = ['ann'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2021 # end analysis year

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]

for se in lse:
    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [dwt2m,gr] = pickle.load(open('%s/dwidth_%s.%s.%02d.%s.pickle' % (odir,yr,varn,xpc,se), 'rb'))

        # store data
        if c0 == 0:
            ydwt2m = np.empty([len(lyr),len(gr['lat']),len(gr['lon'])])
            c0 = 1

        ydwt2m[iyr,...] = dwt2m

    # regress in time
    sdwt2m = np.empty([len(gr['lat']),len(gr['lon'])]) # slope of regression
    idwt2m = np.empty_like(sdwt2m) # intercept of regression
    rdwt2m = np.empty_like(sdwt2m) # r value of regression
    pdwt2m = np.empty_like(sdwt2m) # p value of regression
    esdwt2m = np.empty_like(sdwt2m) # standard error of slope
    eidwt2m = np.empty_like(sdwt2m) # standard error of intercept 
    for ila in tqdm(range(len(gr['lat']))):
        for ilo in range(len(gr['lon'])):
            lrr = linregress(tyr,ydwt2m[:,ila,ilo])
            sdwt2m[ila,ilo] = lrr.slope
            idwt2m[ila,ilo] = lrr.intercept
            rdwt2m[ila,ilo] = lrr.rvalue
            pdwt2m[ila,ilo] = lrr.pvalue
            esdwt2m[ila,ilo] = lrr.stderr
            eidwt2m[ila,ilo] = lrr.intercept_stderr

    stats={}
    stats['slope'] = sdwt2m
    stats['intercept'] = idwt2m
    stats['rvalue'] = rdwt2m
    stats['pvalue'] = pdwt2m
    stats['stderr'] = esdwt2m
    stats['intercept_stderr'] = eidwt2m

    pickle.dump([stats, gr], open('%s/regress.dw.%s.%02d.%s.pickle' % (odir,varn,xpc,se), 'wb'), protocol=5)	
