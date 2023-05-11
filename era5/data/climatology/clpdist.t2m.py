import os,sys
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm

# computes the distance between x percentile and 50th percentile values for the climatology

varn='t2m'
lxpc=[95,99] # percentile value from which to take the distance of the median
# choose from:
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann'] # season (ann, djf, mam, jja, son)
# byr = [1959,2020] # begin analysis year
byr = [1980,2000] # begin analysis year

for xpc in lxpc:
    if xpc==95:
        ixpc=1
    elif xpc==99:
        ixpc=2

    for se in lse:
        odir='/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)
        if not os.path.exists(odir):
            os.makedirs(odir)

        # load data
        c0=0 # first loop counter
        [hvar, gr] = pickle.load(open('%s/c%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'rb'))
        dhvar = hvar[ixpc,...]-hvar[0,...]

        pickle.dump([dhvar, gr], open('%s/cldist.%s.%02d.%g-%g.%s.pickle' % (odir,varn,xpc,byr[0],byr[1],se), 'wb'), protocol=5)	
