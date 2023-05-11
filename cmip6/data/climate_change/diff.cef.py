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

varn='ef'
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

            [hef0, gr] = pickle.load(open('%s/cef_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [hef1, gr] = pickle.load(open('%s/cef_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            defr=hef1-hef0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idefr={}
                c0 = 1

            igr[md]=gr
            idefr[md]=defr

            for ipc in range(len(lpc)):
                defrpc=defr[ipc,...] # save per percentile

                pickle.dump([defrpc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        edefr=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            edefr[imd,...]=idefr[md]

        # # regrid everything to CESM grid
        # edefr=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        # for imd in tqdm(range(len(lmd))):
        #     md=lmd[imd]
        #     if md!='CESM2':
        #         fint=interp1d(igr[md]['lat'],idefr[md],axis=1,fill_value='extrapolate')
        #         lati=fint(egr['lat'])
        #         fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
        #         edefr[imd,...]=fint(egr['lon'])
        #     else:
        #         edefr[imd,...]=idefr[md]

        # compute ensemble statistics
        avgef = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdefr = np.empty_like(avgef) # ensemble standard deviation
        p50ef = np.empty_like(avgef) # ensemble median
        p25ef = np.empty_like(avgef) # ensemble 25th prc
        p75ef = np.empty_like(avgef) # ensemble 75th prc
        iqref = np.empty_like(avgef) # ensemble IQR
        minef = np.empty_like(avgef) # ensemble minimum
        maxef = np.empty_like(avgef) # ensemble maximum
        ptpef = np.empty_like(avgef) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgef[ila,ilo] = np.mean(      edefr[:,ipc,ila,ilo],axis=0)
                    stdefr[ila,ilo] = np.std(       edefr[:,ipc,ila,ilo],axis=0)
                    p50ef[ila,ilo] = np.percentile(edefr[:,ipc,ila,ilo],50,axis=0)
                    p25ef[ila,ilo] = np.percentile(edefr[:,ipc,ila,ilo],25,axis=0)
                    p75ef[ila,ilo] = np.percentile(edefr[:,ipc,ila,ilo],75,axis=0)
                    minef[ila,ilo] = np.amin(      edefr[:,ipc,ila,ilo],axis=0)
                    maxef[ila,ilo] = np.amax(      edefr[:,ipc,ila,ilo],axis=0)
                    ptpef[ila,ilo] = np.ptp(       edefr[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgef
            stats['stdev'] = stdefr
            stats['median'] = p50ef
            stats['prc25'] = p25ef
            stats['prc75'] = p75ef
            stats['min'] = minef
            stats['max'] = maxef
            stats['range'] = ptpef

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

