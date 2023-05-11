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

varn='mrsos'
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

            [hmrsos0, gr] = pickle.load(open('%s/cmrsos_%s.%s.rg.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [hmrsos1, gr] = pickle.load(open('%s/cmrsos_%s.%s.rg.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            dmrsos=hmrsos1-hmrsos0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idmrsos={}
                c0 = 1

            igr[md]=gr
            idmrsos[md]=dmrsos

            for ipc in range(len(lpc)):
                dmrsospc=dmrsos[ipc,...] # save per percentile

                pickle.dump([dmrsospc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        # regrid everything to CESM grid
        edmrsos=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            if md!='CESM2':
                fint=interp1d(igr[md]['lat'],idmrsos[md],axis=1,fill_value='extrapolate')
                lati=fint(egr['lat'])
                fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
                edmrsos[imd,...]=fint(egr['lon'])
            else:
                edmrsos[imd,...]=idmrsos[md]

        # compute ensemble statistics
        avgmrsos = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdmrsos = np.empty_like(avgmrsos) # ensemble standard deviation
        p50mrsos = np.empty_like(avgmrsos) # ensemble median
        p25mrsos = np.empty_like(avgmrsos) # ensemble 25th prc
        p75mrsos = np.empty_like(avgmrsos) # ensemble 75th prc
        iqrmrsos = np.empty_like(avgmrsos) # ensemble IQR
        minmrsos = np.empty_like(avgmrsos) # ensemble minimum
        maxmrsos = np.empty_like(avgmrsos) # ensemble maximum
        ptpmrsos = np.empty_like(avgmrsos) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgmrsos[ila,ilo] = np.mean(      edmrsos[:,ipc,ila,ilo],axis=0)
                    stdmrsos[ila,ilo] = np.std(       edmrsos[:,ipc,ila,ilo],axis=0)
                    p50mrsos[ila,ilo] = np.percentile(edmrsos[:,ipc,ila,ilo],50,axis=0)
                    p25mrsos[ila,ilo] = np.percentile(edmrsos[:,ipc,ila,ilo],25,axis=0)
                    p75mrsos[ila,ilo] = np.percentile(edmrsos[:,ipc,ila,ilo],75,axis=0)
                    minmrsos[ila,ilo] = np.amin(      edmrsos[:,ipc,ila,ilo],axis=0)
                    maxmrsos[ila,ilo] = np.amax(      edmrsos[:,ipc,ila,ilo],axis=0)
                    ptpmrsos[ila,ilo] = np.ptp(       edmrsos[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgmrsos
            stats['stdev'] = stdmrsos
            stats['median'] = p50mrsos
            stats['prc25'] = p25mrsos
            stats['prc75'] = p75mrsos
            stats['min'] = minmrsos
            stats['max'] = maxmrsos
            stats['range'] = ptpmrsos

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

