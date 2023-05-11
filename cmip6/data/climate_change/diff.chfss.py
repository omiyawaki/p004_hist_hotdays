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

varn='hfss'
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

            [hhfss0, gr] = pickle.load(open('%s/chfss_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [hhfss1, gr] = pickle.load(open('%s/chfss_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            dhfss=hhfss1-hhfss0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idhfss={}
                c0 = 1

            igr[md]=gr
            idhfss[md]=dhfss

            for ipc in range(len(lpc)):
                dhfsspc=dhfss[ipc,...] # save per percentile

                pickle.dump([dhfsspc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        edhfss=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            edhfss[imd,...]=idhfss[md]

        # # regrid everything to CESM grid
        # edhfss=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        # for imd in tqdm(range(len(lmd))):
        #     md=lmd[imd]
        #     if md!='CESM2':
        #         fint=interp1d(igr[md]['lat'],idhfss[md],axis=1,fill_value='extrapolate')
        #         lati=fint(egr['lat'])
        #         fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
        #         edhfss[imd,...]=fint(egr['lon'])
        #     else:
        #         edhfss[imd,...]=idhfss[md]

        # compute ensemble statistics
        avghfss = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdhfss = np.empty_like(avghfss) # ensemble standard deviation
        p50hfss = np.empty_like(avghfss) # ensemble median
        p25hfss = np.empty_like(avghfss) # ensemble 25th prc
        p75hfss = np.empty_like(avghfss) # ensemble 75th prc
        iqrhfss = np.empty_like(avghfss) # ensemble IQR
        minhfss = np.empty_like(avghfss) # ensemble minimum
        maxhfss = np.empty_like(avghfss) # ensemble maximum
        ptphfss = np.empty_like(avghfss) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avghfss[ila,ilo] = np.mean(      edhfss[:,ipc,ila,ilo],axis=0)
                    stdhfss[ila,ilo] = np.std(       edhfss[:,ipc,ila,ilo],axis=0)
                    p50hfss[ila,ilo] = np.percentile(edhfss[:,ipc,ila,ilo],50,axis=0)
                    p25hfss[ila,ilo] = np.percentile(edhfss[:,ipc,ila,ilo],25,axis=0)
                    p75hfss[ila,ilo] = np.percentile(edhfss[:,ipc,ila,ilo],75,axis=0)
                    minhfss[ila,ilo] = np.amin(      edhfss[:,ipc,ila,ilo],axis=0)
                    maxhfss[ila,ilo] = np.amax(      edhfss[:,ipc,ila,ilo],axis=0)
                    ptphfss[ila,ilo] = np.ptp(       edhfss[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avghfss
            stats['stdev'] = stdhfss
            stats['median'] = p50hfss
            stats['prc25'] = p25hfss
            stats['prc75'] = p75hfss
            stats['min'] = minhfss
            stats['max'] = maxhfss
            stats['range'] = ptphfss

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

