#!/glade/work/miyawaki/conda-envs/g/bin/python
#PBS -l select=1:ncpus=1:mpiprocs=1
#PBS -l walltime=06:00:00
#PBS -q regular 
#PBS -A P54048000
#PBS -N d-xaaer-loop

import os
import sys
sys.path.append('../')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import interp1d
from tqdm import tqdm
from cmip6util import mods,simu

# collect warmings across the ensembles

lfo = ['ssp245'] # forcing (e.g., ssp245)
# lse = ['ann'] # season (ann, djf, mam, jja, son)
lse = ['ann','jja','son','djf','mam'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1980-2000' # historical analysis period
fut = '2080-2100' # future analysis period

lpc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # available percentiles

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        c0=0 # first loop counter
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            # function that returns directory names where files are located
            def mdir(se,cl,fo,md):
                rdir = '/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s' % (se,cl,fo,md)
                return rdir

            odir=mdir(se,cl,fo,md)
            if not os.path.exists(odir):
                os.makedirs(odir)

            [ht2m0, gr] = pickle.load(open('%s/ht2m_%s.%s.pickle' % (mdir(se,'his','historical',md),his,se), 'rb'))
            [ht2m1, gr] = pickle.load(open('%s/ht2m_%s.%s.pickle' % (mdir(se,'fut',fo,md),fut,se), 'rb'))
            dt2m=ht2m1-ht2m0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idt2m={}
                c0 = 1

            igr[md]=gr
            idt2m[md]=dt2m

            for ipc in range(len(lpc)):
                dt2mpc=dt2m[ipc,...] # save per percentile

                pickle.dump([dt2mpc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2-WACCM':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm')
        if not os.path.exists(edir):
            os.makedirs(edir)

        # regrid everything to CESM grid
        edt2m=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            if md!='CESM2-WACCM':
                fint=interp1d(igr[md]['lat'],idt2m[md],axis=1,fill_value='extrapolate')
                lati=fint(egr['lat'])
                fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
                edt2m[imd,...]=fint(egr['lon'])
            else:
                edt2m[imd,...]=idt2m[md]

        # compute ensemble statistics
        avgt2m = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdt2m = np.empty_like(avgt2m) # ensemble standard deviation
        p50t2m = np.empty_like(avgt2m) # ensemble median
        p25t2m = np.empty_like(avgt2m) # ensemble 25th prc
        p75t2m = np.empty_like(avgt2m) # ensemble 75th prc
        iqrt2m = np.empty_like(avgt2m) # ensemble IQR
        mint2m = np.empty_like(avgt2m) # ensemble minimum
        maxt2m = np.empty_like(avgt2m) # ensemble maximum
        ptpt2m = np.empty_like(avgt2m) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgt2m[ila,ilo] = np.mean(      edt2m[:,ipc,ila,ilo],axis=0)
                    stdt2m[ila,ilo] = np.std(       edt2m[:,ipc,ila,ilo],axis=0)
                    p50t2m[ila,ilo] = np.percentile(edt2m[:,ipc,ila,ilo],50,axis=0)
                    p25t2m[ila,ilo] = np.percentile(edt2m[:,ipc,ila,ilo],25,axis=0)
                    p75t2m[ila,ilo] = np.percentile(edt2m[:,ipc,ila,ilo],75,axis=0)
                    mint2m[ila,ilo] = np.amin(      edt2m[:,ipc,ila,ilo],axis=0)
                    maxt2m[ila,ilo] = np.amax(      edt2m[:,ipc,ila,ilo],axis=0)
                    ptpt2m[ila,ilo] = np.ptp(       edt2m[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgt2m
            stats['stdev'] = stdt2m
            stats['median'] = p50t2m
            stats['prc25'] = p25t2m
            stats['prc75'] = p75t2m
            stats['min'] = mint2m
            stats['max'] = maxt2m
            stats['range'] = ptpt2m

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

