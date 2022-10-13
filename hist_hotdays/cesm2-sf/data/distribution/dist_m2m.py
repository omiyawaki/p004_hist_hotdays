#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -q casper
#PBS -A P54048000
#PBS -N xaaer-loop

import os
import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint as td2q
from tqdm import tqdm
from sfutil import emem,conf,simu,sely
import constants as c

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

lfo = ['lens'] # forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
lse = ['ann'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr_his=[1920,1940] # output year bounds
byr_fut=[2030,2050]

# percentiles to compute (follows Byrne [2021])
pc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] 

for se in lse:
    for fo in lfo:
        for cl in lcl:
            if fo=='lens':
                idir='/project/mojave/cesm2/LENS/atm/day_1'
                odir='/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
            else:
                idir='/glade/campaign/cesm/collections/CESM2-SF/timeseries/atm/proc/tseries/day_1'
                odir='/glade/work/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # list of ensemble member numbers
            lmem=emem(fo)

            # years and simulation names
            if cl == 'fut':
                cnf=conf(fo,cl)
                sim=simu(fo,cl)
                lyr=sely(fo,cl)
                byr=byr_fut
            elif cl == 'his':
                cnf=conf(fo,cl)
                sim=simu(fo,cl)
                lyr=sely(fo,cl)
                byr=byr_his

            for imem in tqdm(range(len(lmem))):
                mem=lmem[imem]
                n=0 # counter
                for yr in lyr:
                    e={} # create empty list to store energy data
                    for varn in ['TREFHT','QREFHT']:
                        if fo=='lens':
                            fn = '%s/%s/b.e21.%s.f09_g17.%s.cam.h1.%s.%s.nc' % (idir,varn,cnf,sim[imem],varn,yr)
                        else:
                            fn = '%s/%s/b.e21.%s.f09_g17.%s.%s.cam.h1.%s.%s.nc' % (idir,varn,cnf,sim,mem,varn,yr)

                        ds = xr.open_dataset(fn)
                        e[varn]=ds[varn]
                    mse=c.cpd*e['TREFHT']*units.kelvin+c.Lv*e['QREFHT']
                    if n==0:
                       m2m=mse
                    else:
                        m2m = xr.concat((m2m,mse),'time')
                    n=n+1
                    
                # select data within time of interest
                m2m=m2m.sel(time=m2m['time.year']>=byr[0])
                m2m=m2m.sel(time=m2m['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    m2m=m2m.sel(time=m2m['time.season']==se.upper())
                
                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']

                # initialize array to store histogram data
                hm2m = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in range(gr['lat'].size):
                        lt = m2m[:,la,ln]
                        hm2m[:,la,ln]=np.percentile(lt,pc)	

                pickle.dump([hm2m, gr], open('%s/hm2m_%g-%g.%s.%s.pickle' % (odir,byr[0],byr[1],mem,se), 'wb'), protocol=5)	
