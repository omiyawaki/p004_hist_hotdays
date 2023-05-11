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

lse = ['djf'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lfo = ['historical'] # forcing (e.g., ssp245)
lcl = ['his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
# lfo = ['ssp370'] # forcing (e.g., ssp245)
# lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
# lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr_his=[1980,2000] # output year bounds
byr_fut=[2080,2100]

# percentiles to compute (follows Byrne [2021])
pc = [1,5,50,95,99] 

for se in lse:
    for fo in lfo:
        for cl in lcl:
            # list of models
            lmd=mods(fo)

            # years and simulation names
            if cl == 'fut':
                byr=byr_fut
            elif cl == 'his':
                byr=byr_his

            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                ens=emem(md)
                sim=simu(fo,cl,None)
                grd=grid(fo,cl,md)
                # lyr=year(cl,md,byr)

                idir='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (sim,freq,varn,md,ens,grd)
                odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                c=0 # counter
                # for yr in lyr:
                #     fn = '%s/%s_%s_%s_%s_%s_%s_%s.nc' % (idir,varn,freq,md,sim,ens,grd,yr)

                #     ds = xr.open_dataset(fn)
                #     if c==0:
                #         t2m = ds[varn]
                #     else:
                #         t2m = xr.concat((t2m,ds[varn]),'time')
                #     c=c+1
                fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idir,varn,freq,md,sim,ens,grd)
                ds = xr.open_mfdataset(fn)
                t2m=ds[varn].load()
                    
                # select data within time of interest
                t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                t2m=t2m.sel(time=t2m['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    t2m=t2m.sel(time=t2m['time.season']==se.upper())
                
                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']

                # initialize array to store histogram data
                ht2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in range(gr['lat'].size):
                        lt = t2m[:,la,ln]
                        ht2m[:,la,ln]=np.percentile(lt,pc)	

                pickle.dump([ht2m, gr], open('%s/h%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'wb'), protocol=5)	
