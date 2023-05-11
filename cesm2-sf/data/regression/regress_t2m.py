import os
import sys
sys.path.append('../')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from sfutil import emem

lse = ['jja'] # season (ann, djf, mam, jja, son)
lcl=['tseries']
lfo=['lens']
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2020 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

lpc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # available percentiles

for se in lse:
    for fo in lfo:
        for cl in lcl:
            odir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)

            # list of ensemble member numbers
            lmem=emem(fo)

            for imem in tqdm(range(len(lmem))):
                mem=lmem[imem]

                # load data
                c0=0 # first loop counter
                for iyr in tqdm(range(len(lyr))):
                    yr = lyr[iyr]
                    [ht2m, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (odir,yr,mem,se), 'rb'))

                    # store data
                    if c0 == 0:
                        yht2m = np.empty([len(lyr),len(lpc),len(gr['lat']),len(gr['lon'])])
                        c0 = 1

                    yht2m[iyr,...] = ht2m

                # regress in time
                sht2m = np.empty([ht2m.shape[1],ht2m.shape[2]]) # slope of regression
                iht2m = np.empty_like(sht2m) # intercept of regression
                rht2m = np.empty_like(sht2m) # r value of regression
                pht2m = np.empty_like(sht2m) # p value of regression
                esht2m = np.empty_like(sht2m) # standard error of slope
                eiht2m = np.empty_like(sht2m) # standard error of intercept 
                for ipc in tqdm(range(len(lpc))):
                    for ilo in tqdm(range(len(gr['lon']))):
                        for ila in range(len(gr['lat'])):
                            lrr = linregress(tyr,yht2m[:,ipc,ila,ilo])
                            sht2m[ila,ilo] = lrr.slope
                            iht2m[ila,ilo] = lrr.intercept
                            rht2m[ila,ilo] = lrr.rvalue
                            pht2m[ila,ilo] = lrr.pvalue
                            esht2m[ila,ilo] = lrr.stderr
                            eiht2m[ila,ilo] = lrr.intercept_stderr

                    stats={}
                    stats['slope'] = sht2m
                    stats['intercept'] = iht2m
                    stats['rvalue'] = rht2m
                    stats['pvalue'] = pht2m
                    stats['stderr'] = esht2m
                    stats['intercept_stderr'] = eiht2m

                    pickle.dump([stats, gr], open('%s/regress_%02d.%s.%s.pickle' % (odir,lpc[ipc],mem,se), 'wb'), protocol=5)	
