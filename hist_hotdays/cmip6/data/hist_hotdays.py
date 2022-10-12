#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -q regular 
#PBS -A P54048000
#PBS -N fut-ann

import os
import sys
sys.path.append('.')
sys.path.append('/glade/u/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,emem,simu,year
from glade_utils import grid,mctr,adte
import time

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon
varn='tas' # variable name

lfo = ['ssp370'] # forcing (e.g., ssp245)
lse = ['ann'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
# lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr_his=[1920,1940] # output year bounds
byr_fut=[2030,2050]

# percentiles to compute (follows Byrne [2021])
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] 

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
                ctr=mctr(md)
                ens=emem(md)
                dte=adte(fo,cl,md)
                sim=simu(fo,cl)
                grd=grid(fo,cl,md)
                lyr=year(cl,md,byr)

                if ctr=='NCAR':
                    idir='/glade/collections/cmip/CMIP6/ScenarioMIP/%s/%s/%s/%s/%s/%s/%s/%s' % (ctr,md,fo,ens,freq,varn,grd,dte)
                else:
                    idir='/glade/collections/cmip/CMIP6/ScenarioMIP/%s/%s/%s/%s/%s/%s/%s/%s/%s' % (ctr,md,fo,ens,freq,varn,grd,dte,varn)
                odir='/glade/work/miyawaki/data/p004/hist_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (varn,se,cl,fo,md)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                c=0 # counter
                for yr in lyr:
                    fn = '%s/%s_%s_%s_%s_%s_%s_%s.nc' % (idir,varn,freq,md,sim,ens,grd,yr)

                    ds = xr.open_dataset(fn)
                    if c==0:
                        t2m = ds[varn]
                        # save grid info
                        gr = {}
                        gr['lon'] = ds['lon']
                        gr['lat'] = ds['lat']
                    else:
                        t2m = xr.concat((t2m,ds[varn]),'time')
                    ds.close()
                    c=c+1

                # select data within time of interest
                t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                t2m=t2m.sel(time=t2m['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    t2m=t2m.sel(time=t2m['time.season']==se.upper())
                
                # initialize array to store histogram data
                ht2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in tqdm(range(gr['lat'].size)):
                        lt = t2m[:,la,ln]
                        ht2m[:,la,ln]=np.percentile(lt,pc)	

                pickle.dump([ht2m, gr], open('%s/h%s_%g-%g.%s.%s.pickle' % (varn,odir,byr[0],byr[1],se), 'wb'), protocol=5)	
