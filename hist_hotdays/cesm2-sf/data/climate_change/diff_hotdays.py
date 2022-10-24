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
from tqdm import tqdm
from sfutil import emem

# collect warmings across the ensembles

lfo = ['lens'] # forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','mam','jja','son','djf'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1950-1970' # historical analysis period
fut = '2000-2014' # future analysis period
# fut = '2030-2050' # future analysis period

lpc = [1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99] # available percentiles

for fo in lfo:
    for se in lse:
        lmem=emem(fo) # create list of ensemble members

        # function that returns directory names where files are located
        def mdir(se,cl,fo):
            if fo=='lens':
                rdir='/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
            else:
                rdir = '/glade/work/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
            return rdir

        odir=mdir(se,cl,fo)
        if not os.path.exists(odir):
            os.makedirs(odir)

        # load data
        c0=0 # first loop counter
        for imem in tqdm(range(len(lmem))):
            mem = lmem[imem]
            [ht2m0, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'his',fo),his,mem,se), 'rb'))
            if fut=='2000-2014':
                [ht2m1, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'his',fo),fut,mem,se), 'rb'))
            else:
                [ht2m1, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'fut',fo),fut,mem,se), 'rb'))
            dt2m=ht2m1-ht2m0 # take difference

            # store data
            if c0 == 0:
                edt2m = np.empty([len(lmem),len(lpc),len(gr['lat']),len(gr['lon'])])
                c0 = 1

            edt2m[imem,...] = dt2m

        # compute ensemble statistics
        avgt2m = np.empty([len(gr['lat']),len(gr['lon'])]) # ensemble average
        stdt2m = np.empty_like(avgt2m) # ensemble standard deviation
        p50t2m = np.empty_like(avgt2m) # ensemble median
        p25t2m = np.empty_like(avgt2m) # ensemble 25th prc
        p75t2m = np.empty_like(avgt2m) # ensemble 75th prc
        iqrt2m = np.empty_like(avgt2m) # ensemble IQR
        mint2m = np.empty_like(avgt2m) # ensemble minimum
        maxt2m = np.empty_like(avgt2m) # ensemble maximum
        ptpt2m = np.empty_like(avgt2m) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(gr['lon']))):
                for ila in range(len(gr['lat'])):
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

            pickle.dump([stats, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

