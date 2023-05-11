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

varn='tas'
lfo = ['ssp370'] # forcing (e.g., ssp245)
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','jja','son','djf','mam'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1980-2000' # historical analysis period
fut = '2080-2100' # future analysis period

lpc = [1,5,50,95,99] # available percentiles

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

            [htas0, gr] = pickle.load(open('%s/htas_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),his,se), 'rb'))
            [htas1, gr] = pickle.load(open('%s/htas_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),fut,se), 'rb'))
            dtas=htas1-htas0 # take difference

            # store data
            if c0 == 0:
                igr={}
                idtas={}
                c0 = 1

            igr[md]=gr
            idtas[md]=dtas

            for ipc in range(len(lpc)):
                dtaspc=dtas[ipc,...] # save per percentile

                pickle.dump([dtaspc, gr], open('%s/diff_%02d.%s.%s.%s.pickle' % (odir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

            # save CESM grid for ensemble mean
            if md=='CESM2-WACCM':
                egr=gr


        # ENSEMBLE 
        # output directory
        edir=mdir(se,cl,fo,'mmm',varn)
        if not os.path.exists(edir):
            os.makedirs(edir)

        # regrid everything to CESM grid
        edtas=np.empty([len(lmd),len(lpc),len(egr['lat']),len(egr['lon'])])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            if md!='CESM2-WACCM':
                fint=interp1d(igr[md]['lat'],idtas[md],axis=1,fill_value='extrapolate')
                lati=fint(egr['lat'])
                fint=interp1d(igr[md]['lon'],lati,axis=2,fill_value='extrapolate')
                edtas[imd,...]=fint(egr['lon'])
            else:
                edtas[imd,...]=idtas[md]

        # compute ensemble statistics
        avgtas = np.empty([len(egr['lat']),len(egr['lon'])]) # ensemble average
        stdtas = np.empty_like(avgtas) # ensemble standard deviation
        p50tas = np.empty_like(avgtas) # ensemble median
        p25tas = np.empty_like(avgtas) # ensemble 25th prc
        p75tas = np.empty_like(avgtas) # ensemble 75th prc
        iqrtas = np.empty_like(avgtas) # ensemble IQR
        mintas = np.empty_like(avgtas) # ensemble minimum
        maxtas = np.empty_like(avgtas) # ensemble maximum
        ptptas = np.empty_like(avgtas) # ensemble range
        for ipc in tqdm(range(len(lpc))):
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgtas[ila,ilo] = np.mean(      edtas[:,ipc,ila,ilo],axis=0)
                    stdtas[ila,ilo] = np.std(       edtas[:,ipc,ila,ilo],axis=0)
                    p50tas[ila,ilo] = np.percentile(edtas[:,ipc,ila,ilo],50,axis=0)
                    p25tas[ila,ilo] = np.percentile(edtas[:,ipc,ila,ilo],25,axis=0)
                    p75tas[ila,ilo] = np.percentile(edtas[:,ipc,ila,ilo],75,axis=0)
                    mintas[ila,ilo] = np.amin(      edtas[:,ipc,ila,ilo],axis=0)
                    maxtas[ila,ilo] = np.amax(      edtas[:,ipc,ila,ilo],axis=0)
                    ptptas[ila,ilo] = np.ptp(       edtas[:,ipc,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgtas
            stats['stdev'] = stdtas
            stats['median'] = p50tas
            stats['prc25'] = p25tas
            stats['prc75'] = p75tas
            stats['min'] = mintas
            stats['max'] = maxtas
            stats['range'] = ptptas

            pickle.dump([stats, egr], open('%s/diff_%02d.%s.%s.%s.pickle' % (edir,lpc[ipc],his,fut,se), 'wb'), protocol=5)	

