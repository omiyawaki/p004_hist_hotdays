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

varn='hfls'
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

            [hhfls0, gr] = pickle.load(open('%s/chfls_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [hhfls1, gr] = pickle.load(open('%s/chfls_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            dhfls=hhfls1-hhfls0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idhfls={}
                c0 = 1

            igr[md]=gr
            idhfls[md]=dhfls

            for ipc in range(len(lpc)):
                dhflspc=dhfls[ipc,...] # save per percentile

                pickle.dump([dhflspc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        edhfls=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            edhfls[imd,...]=idhfls[md]

        # # regrid everything to CESM grid
        # edhfls=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        # for imd in tqdm(range(len(lmd))):
        #     md=lmd[imd]
        #     if md!='CESM2':
        #         fint=interp1d(igr[md]['lat'],idhfls[md],axis=1,fill_value='extrapolate')
        #         lati=fint(egr['lat'])
        #         fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
        #         edhfls[imd,...]=fint(egr['lon'])
        #     else:
        #         edhfls[imd,...]=idhfls[md]

        # compute ensemble statistics
        avghfls = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdhfls = np.empty_like(avghfls) # ensemble standard deviation
        p50hfls = np.empty_like(avghfls) # ensemble median
        p25hfls = np.empty_like(avghfls) # ensemble 25th prc
        p75hfls = np.empty_like(avghfls) # ensemble 75th prc
        iqrhfls = np.empty_like(avghfls) # ensemble IQR
        minhfls = np.empty_like(avghfls) # ensemble minimum
        maxhfls = np.empty_like(avghfls) # ensemble maximum
        ptphfls = np.empty_like(avghfls) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avghfls[ila,ilo] = np.mean(      edhfls[:,ipc,ila,ilo],axis=0)
                    stdhfls[ila,ilo] = np.std(       edhfls[:,ipc,ila,ilo],axis=0)
                    p50hfls[ila,ilo] = np.percentile(edhfls[:,ipc,ila,ilo],50,axis=0)
                    p25hfls[ila,ilo] = np.percentile(edhfls[:,ipc,ila,ilo],25,axis=0)
                    p75hfls[ila,ilo] = np.percentile(edhfls[:,ipc,ila,ilo],75,axis=0)
                    minhfls[ila,ilo] = np.amin(      edhfls[:,ipc,ila,ilo],axis=0)
                    maxhfls[ila,ilo] = np.amax(      edhfls[:,ipc,ila,ilo],axis=0)
                    ptphfls[ila,ilo] = np.ptp(       edhfls[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avghfls
            stats['stdev'] = stdhfls
            stats['median'] = p50hfls
            stats['prc25'] = p25hfls
            stats['prc75'] = p75hfls
            stats['min'] = minhfls
            stats['max'] = maxhfls
            stats['range'] = ptphfls

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

