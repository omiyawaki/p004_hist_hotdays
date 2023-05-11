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

varn='huss'
lfo = ['ssp370'] # forcing (e.g., ssp245)
lse = ['jja'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1980-2000' # historical analysis period
fut = '2080-2100' # future analysis period

lpc = [0,95,99] # available percentiles

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        c0=0 # first loop counter
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            # function that returns directory names where files are located
            def mdir(se,cl,fo,md,varn):
                rdir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
                return rdir

            odir=mdir(se,cl,fo,md,varn)
            if not os.path.exists(odir):
                os.makedirs(odir)

            [hhuss0, gr] = pickle.load(open('%s/chuss_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [hhuss1, gr] = pickle.load(open('%s/chuss_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            dhuss=hhuss1-hhuss0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idhuss={}
                c0 = 1

            igr[md]=gr
            idhuss[md]=dhuss

            for ipc in range(len(lpc)):
                dhusspc=dhuss[ipc,...] # save per percentile

                pickle.dump([dhusspc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        # regrid everything to CESM grid
        edhuss=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            if md!='CESM2':
                fint=interp1d(igr[md]['lat'],idhuss[md],axis=1,fill_value='extrapolate')
                lati=fint(egr['lat'])
                fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
                edhuss[imd,...]=fint(egr['lon'])
            else:
                edhuss[imd,...]=idhuss[md]

        # compute ensemble statistics
        avghuss = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdhuss = np.empty_like(avghuss) # ensemble standard deviation
        p50huss = np.empty_like(avghuss) # ensemble median
        p25huss = np.empty_like(avghuss) # ensemble 25th prc
        p75huss = np.empty_like(avghuss) # ensemble 75th prc
        iqrhuss = np.empty_like(avghuss) # ensemble IQR
        minhuss = np.empty_like(avghuss) # ensemble minimum
        maxhuss = np.empty_like(avghuss) # ensemble maximum
        ptphuss = np.empty_like(avghuss) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avghuss[ila,ilo] = np.mean(      edhuss[:,ipc,ila,ilo],axis=0)
                    stdhuss[ila,ilo] = np.std(       edhuss[:,ipc,ila,ilo],axis=0)
                    p50huss[ila,ilo] = np.percentile(edhuss[:,ipc,ila,ilo],50,axis=0)
                    p25huss[ila,ilo] = np.percentile(edhuss[:,ipc,ila,ilo],25,axis=0)
                    p75huss[ila,ilo] = np.percentile(edhuss[:,ipc,ila,ilo],75,axis=0)
                    minhuss[ila,ilo] = np.amin(      edhuss[:,ipc,ila,ilo],axis=0)
                    maxhuss[ila,ilo] = np.amax(      edhuss[:,ipc,ila,ilo],axis=0)
                    ptphuss[ila,ilo] = np.ptp(       edhuss[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avghuss
            stats['stdev'] = stdhuss
            stats['median'] = p50huss
            stats['prc25'] = p25huss
            stats['prc75'] = p75huss
            stats['min'] = minhuss
            stats['max'] = maxhuss
            stats['range'] = ptphuss

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

