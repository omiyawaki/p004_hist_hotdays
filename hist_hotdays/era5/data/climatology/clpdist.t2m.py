import os,sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# computes the distance between x percentile and 50th percentile values for the climatology

varn='t2m'
xpc=95 # percentile value from which to take the distance of the median
# choose from:
lpc=[1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99]
ixpc=np.where(np.array(lpc)==xpc)[0][0] # index of xpc:
imed=np.where(np.array(lpc)==50)[0][0] # index of median

# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lse = ['ann'] # season (ann, djf, mam, jja, son)
y0 = 1950 # begin analysis year
y1 = 2021 # end analysis year

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]


for se in lse:
    odir='/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    # load data
    c0=0 # first loop counter
    for iyr in tqdm(range(len(lyr))):
        yr = lyr[iyr]
        [hvar, gr] = pickle.load(open('%s/h%s_%s.%s.pickle' % (odir,varn,yr,se), 'rb'))

        # store data
        if c0 == 0:
            yhvar = np.empty([len(lyr),len(gr['lat']),len(gr['lon'])])
            c0 = 1

        yhvar[iyr,...] = hvar[ixpc,...]-hvar[imed,...]

    # take mean over time
    dhvar=np.mean(yhvar,axis=0)

    pickle.dump([dhvar, gr], open('%s/cldist.%02d.%s.pickle' % (odir,xpc,se), 'wb'), protocol=5)	
