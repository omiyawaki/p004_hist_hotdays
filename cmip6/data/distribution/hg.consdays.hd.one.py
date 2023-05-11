import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from scipy.ndimage import label,sum_labels
from tqdm import tqdm
from cmip6util import mods,emem,simu,year
from glade_utils import grid
np.set_printoptions(threshold=sys.maxsize)

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 
iloc=[110,85] # SEA

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon
varn='hd' # variable name
lse = ['jja'] # season (ann, djf, mam, jja, son)

# fo='historical'
# lcl = ['his']
# byr=[1980,2000] # output year bounds

fo='ssp370'
lcl = ['fut']
byr=[2080,2100] # output year bounds

# percentiles to compute (follows Byrne [2021])
lpc = [95,99] 

for pc in lpc:
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
                fn = '%s/cl%s_%g-%g.%g.%s.nc' % (idir,varn,byr[0],byr[1],pc,se)

                ds = xr.open_mfdataset(fn)
                hd = ds[varn].load()[:,iloc[0],iloc[1]]
                lhd=label(hd)
                dhd=sum_labels(hd,labels=lhd[0],index=np.arange(lhd[1])+1)

                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']

                pickle.dump([dhd,gr], open('%s/cd%s_%g-%g.%g.%s.pickle' % (odir,varn,byr[0],byr[1],pc,se), 'wb'), protocol=5)	
