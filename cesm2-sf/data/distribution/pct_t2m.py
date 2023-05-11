#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -A P54048000
#PBS -N xaaer-loop

import os
import sys
sys.path.append('../')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from sfutil import emem,conf,simu,sely

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

lfo = ['lens'] # forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['djf','mam','son'] # season (ann, djf, mam, jja, son)
lcl = ['tseries'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr=[2021,2100] # output year bounds

# percentiles to compute (follows Byrne [2021])
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] 

for se in lse:
    for fo in lfo:
        for cl in lcl:
            if fo=='lens':
                idir='/project/mojave/cesm2/LENS/atm/day_1/TREFHT'
                odir='/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
            else:
                idir='/glade/campaign/cesm/collections/CESM2-SF/timeseries/atm/proc/tseries/day_1/TREFHT'
                odir='/glade/work/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # list of ensemble member numbers
            lmem=emem(fo)

            # years and simulation names
            lyr=sely(fo,cl)

            for imem in tqdm(range(len(lmem))):
                mem=lmem[imem]

                for yr in tqdm(lyr):
                    yr0=int(yr[0:4])
                    yr1=int(yr[9:13])

                    cnf=conf(fo,cl,yr=yr0)
                    sim=simu(fo,cl,yr=yr0)
                    if fo=='lens':
                        fn = '%s/b.e21.%s.f09_g17.%s.cam.h1.TREFHT.%s.nc' % (idir,cnf,sim[imem],yr)
                    else:
                        fn = '%s/b.e21.%s.f09_g17.%s.%s.cam.h1.TREFHT.%s.nc' % (idir,cnf,sim,mem,yr)

                    ds = xr.open_dataset(fn)
                    t2m = ds['TREFHT'].load()
                    # if c==0:
                    #     t2m = ds['TREFHT']
                    # else:
                    #     t2m = xr.concat((t2m,ds['TREFHT']),'time')
                    
                    # select data within time of interest if necessary
                    if yr0<byr[0]:
                        yr0=byr[0]
                        t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                    if yr1>byr[1]:
                        yr1=byr[1]
                        t2m=t2m.sel(time=t2m['time.year']<=byr[1])

                    # save data year by year
                    for eyr in tqdm(np.arange(yr0,yr1+1)):
                        # select data within time of interest
                        yt2m=t2m.sel(time=t2m['time.year']==eyr)

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

                        pickle.dump([ht2m, gr], open('%s/ht2m_%g.%s.%s.pickle' % (odir,eyr,mem,se), 'wb'), protocol=5)	
