import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,emem,simu,year
from glade_utils import grid

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon
varn='tas' # variable name
lse = ['son','mam'] # season (ann, djf, mam, jja, son)

fo='historical'
lcl = ['his']
byr=[1980,2000] # output year bounds

# fo='ssp370'
# lcl = ['fut']
# byr=[2080,2100] # output year bounds

# percentiles to compute (follows Byrne [2021])
pc = [1,5,50,95,99] 

for cl in lcl:
    for se in lse:
        # list of models
        lmd=mods(fo)

        # years and simulation names
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            c=0 # counter
            fn = '%s/cl%s_%g-%g.%s.nc' % (idir,varn,byr[0],byr[1],se)

            ds = xr.open_mfdataset(fn)
            t2m = ds[varn].load()

            # save grid info
            gr = {}
            gr['lon'] = ds['lon']
            gr['lat'] = ds['lat']
            ds.close()

            # initialize array to store histogram data
            ht2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

            # loop through gridir points to compute percentiles
            for ln in tqdm(range(gr['lon'].size)):
                for la in range(gr['lat'].size):
                    lt = t2m[:,la,ln]
                    ht2m[:,la,ln]=np.percentile(lt,pc)	

            pickle.dump([ht2m, gr], open('%s/ht2m_%g-%g.%s.pickle' % (odir,byr[0],byr[1],se), 'wb'), protocol=5)	
