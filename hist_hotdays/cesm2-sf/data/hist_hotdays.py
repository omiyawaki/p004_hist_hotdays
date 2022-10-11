#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -A P54048000
#PBS -N xaaer-loop

import os
import sys
sys.path.append('.')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from sfutil import emem,conf,simu

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

lfo = ['xaaer'] # forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])

# percentiles to compute (follows Byrne [2021])
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] 

for se in lse:
    for fo in lfo:
        for cl in lcl:
            idir = '/glade/campaign/cesm/collections/CESM2-SF/timeseries/atm/proc/tseries/day_1/TREFHT'
            odir = '/glade/work/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # list of ensemble member numbers
            lmem=emem(fo)

            # years and simulation names
            if cl == 'fut':
                cnf=conf(fo,cl)
                sim=simu(fo,cl)
                lyr=['20250101-20341231', '20350101-20441231', '20450101-20501231'] # future
                byr=[2030,2050] # output year bounds
            elif cl == 'his':
                cnf=conf(fo,cl)
                sim=simu(fo,cl)
                lyr=['19200101-19291231', '19300101-19391231', '19400101-19491231'] # past
                byr=[1920,1940] # output year bounds

            for mem in tqdm(lmem):
                c=0 # counter
                for yr in lyr:
                    fn = '%s/b.e21.%s.f09_g17.%s.%s.cam.h1.TREFHT.%s.nc' % (idir,cnf,sim,mem,yr)
                    ds = xr.open_dataset(fn)
                    if c==0:
                        t2m = ds['TREFHT']
                    else:
                        t2m = xr.concat((t2m,ds['TREFHT']),'time')
                    c=c+1
                    
                # select data within time of interest
                t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                t2m=t2m.sel(time=t2m['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    print(t2m.shape)
                    t2m=t2m.sel(time=t2m['time.season']==se.upper())
                    print(t2m.shape)
                
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

                pickle.dump([ht2m, gr], open('%s/ht2m_%g-%g.%s.%s.pickle' % (odir,byr[0],byr[1],mem,se), 'wb'), protocol=5)	