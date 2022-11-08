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

fo='historical'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['jja','mam','son','djf'] # season (ann, djf, mam, jja, son)
lcl = ['tseries']
byr=[1950,2015] # output year bounds
lyr=np.arange(byr[0],byr[1]+1)

# percentiles to compute (follows Byrne [2021])
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] 

for cl in lcl:
    for se in lse:
        # list of models
        lmd=mods(fo)

        # years and simulation names
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            ens=emem(md)
            grd=grid(fo,cl,md)

            idir='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (fo,freq,varn,md,ens,grd)
            odir='/project/amp/miyawaki/data/p004/hist_hotdays/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)

            if not os.path.exists(odir):
                os.makedirs(odir)

            c=0 # counter
            fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,fo,ens,grd)

            ds = xr.open_mfdataset(fn)
            t2m = ds[varn]

            for yr in tqdm(lyr):
                # select data within time of interest
                t2ms=t2m.sel(time=t2m['time.year']>=byr[0])
                t2ms=t2m.sel(time=t2m['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    t2ms=t2ms.sel(time=t2ms['time.season']==se.upper())
                            
                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']
                ds.close()
                t2ms.load()

                # initialize array to store histogram data
                ht2ms = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in range(gr['lat'].size):
                        lt = t2ms[:,la,ln]
                        ht2ms[:,la,ln]=np.percentile(lt,pc)	

                pickle.dump([ht2ms, gr], open('%s/ht2ms_%g-%g.%s.pickle' % (odir,byr[0],byr[1],se), 'wb'), protocol=5)	
