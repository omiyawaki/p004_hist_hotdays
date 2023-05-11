import sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
varn='t2m'
lre=['sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2021 # end analysis year
bm=np.arange(285,310,0.1) # bins for computing pdf with kde

tyr=np.arange(y0,y1)
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

        # regress in time
        skt2m = np.empty(len(bm)) # slope of regression
        ikt2m = np.empty_like(skt2m) # intercept of regression
        rkt2m = np.empty_like(skt2m) # r value of regression
        pkt2m = np.empty_like(skt2m) # p value of regression
        eskt2m = np.empty_like(skt2m) # standard error of slope
        eikt2m = np.empty_like(skt2m) # standard error of intercept 
        for ibm in tqdm(range(len(bm))):
            lrr = linregress(tyr,ykt2m[:,ibm])
            skt2m[ibm] = lrr.slope
            ikt2m[ibm] = lrr.intercept
            rkt2m[ibm] = lrr.rvalue
            pkt2m[ibm] = lrr.pvalue
            eskt2m[ibm] = lrr.stderr
            eikt2m[ibm] = lrr.intercept_stderr

            stats={}
            stats['slope'] = skt2m
            stats['intercept'] = ikt2m
            stats['rvalue'] = rkt2m
            stats['pvalue'] = pkt2m
            stats['stderr'] = eskt2m
            stats['intercept_stderr'] = eikt2m

            pickle.dump([stats, bm], open('%s/regress.kde.%s.%s.%s.pickle' % (odir,varn,re,se), 'wb'), protocol=5)	
