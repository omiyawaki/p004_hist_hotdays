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

lfo = ['historical'] # forcing (e.g., ssp245)
# lse = ['ann','jja','djf','son','mam'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
cl='his' # climatology (difference between future and historical)
yr = '1980-2000' # historical analysis period

for fo in lfo:
    for se in lse:
        lmd=mods(fo) # create list of ensemble members

        c0=0 # first loop counter
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            # function that returns directory names where files are located
            def mdir(se,cl,fo,md,varn):
                rdir = '/project/amp/miyawaki/data/p004/hist_hotdays/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
                return rdir

            hdir=mdir(se,'his','historical',md,varn)
            fdir=mdir(se,'fut',fo,md,varn)
            odir=mdir(se,cl,fo,md,varn)
            if not os.path.exists(hdir):
                os.makedirs(hdir)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            if not os.path.exists(odir):
                os.makedirs(odir)

            [ht2m0, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (mdir(se,cl,fo,md,varn),varn,yr,se), 'rb'))

            # store data
            if c0 == 0:
                igr={}
                iht2m0={}
                c0 = 1

            igr[md]=gr
            iht2m0[md]=ht2m0

            # save CESM grid for ensemble mean
            if md=='CESM2':
                egr=gr


        # ENSEMBLE 
        # output directory
        midir=mdir(se,cl,fo,'mi',varn)
        if not os.path.exists(midir):
            os.makedirs(midir)
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        # regrid everything to CESM grid
        eht2m0=np.empty([len(lmd),3,len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            if md!='CESM2':
                fint=interp1d(igr[md]['lat'],iht2m0[md],axis=1,fill_value='extrapolate')
                lati=fint(egr['lat'])
                fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
                eht2m0[imd,...]=fint(egr['lon'])

            else:
                eht2m0[imd,...]=iht2m0[md]

        pickle.dump([eht2m0, md, egr], open('%s/cl%s_%s.%s.pickle' % (midir,varn,yr,se), 'wb'), protocol=5)	

        # compute ensemble statistics
        avgt2m = np.empty([3,len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdt2m = np.empty_like(avgt2m) # ensemble standard deviation
        p50t2m = np.empty_like(avgt2m) # ensemble median
        p25t2m = np.empty_like(avgt2m) # ensemble 25th prc
        p75t2m = np.empty_like(avgt2m) # ensemble 75th prc
        iqrt2m = np.empty_like(avgt2m) # ensemble IQR
        mint2m = np.empty_like(avgt2m) # ensemble minimum
        maxt2m = np.empty_like(avgt2m) # ensemble maximum
        ptpt2m = np.empty_like(avgt2m) # ensemble range
        for ipc in tqdm(range(3)):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgt2m[ipc,ila,ilo] = np.mean(      eht2m0[:,ipc,ila,ilo],axis=0)
                    stdt2m[ipc,ila,ilo] = np.std(       eht2m0[:,ipc,ila,ilo],axis=0)
                    p50t2m[ipc,ila,ilo] = np.percentile(eht2m0[:,ipc,ila,ilo],50,axis=0)
                    p25t2m[ipc,ila,ilo] = np.percentile(eht2m0[:,ipc,ila,ilo],25,axis=0)
                    p75t2m[ipc,ila,ilo] = np.percentile(eht2m0[:,ipc,ila,ilo],75,axis=0)
                    mint2m[ipc,ila,ilo] = np.amin(      eht2m0[:,ipc,ila,ilo],axis=0)
                    maxt2m[ipc,ila,ilo] = np.amax(      eht2m0[:,ipc,ila,ilo],axis=0)
                    ptpt2m[ipc,ila,ilo] = np.ptp(       eht2m0[:,ipc,ila,ilo],axis=0)

        stats={}
        stats['mean'] = avgt2m
        stats['stdev'] = stdt2m
        stats['median'] = p50t2m
        stats['prc25'] = p25t2m
        stats['prc75'] = p75t2m
        stats['min'] = mint2m
        stats['max'] = maxt2m
        stats['range'] = ptpt2m

        pickle.dump([stats, egr], open('%s/c%s_%s.%s.pickle' % (edir,varn,yr,se), 'wb'), protocol=5)	
