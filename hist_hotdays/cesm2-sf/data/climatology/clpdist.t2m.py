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
from sfutil import emem

# collect warmings across the ensembles

varn='t2m'
lfo=['lens']
lxpc=[95,1,5,99] # percentile value from which to take the distance of the median
# choose from:
lpc=[1e-3,1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,82,85,87,90,92,95,97,99]
imed=np.where(np.array(lpc)==50)[0][0] # index of median

lse = ['jja'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1980-2000' # historical analysis period
fut = '2080-2100' # future analysis period

for xpc in lxpc:
    ixpc=np.where(np.array(lpc)==xpc)[0][0] # index of xpc:
    for fo in lfo:
        for se in lse:
            lmem=emem(fo) # create list of ensemble members

            def mdir(se,cl,fo):
                rdir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s' % (se,cl,fo)
                return rdir

            hdir=mdir(se,'his',fo)
            fdir=mdir(se,'fut',fo)
            odir=mdir(se,cl,fo)
            if not os.path.exists(hdir):
                os.makedirs(hdir)
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            if not os.path.exists(odir):
                os.makedirs(odir)

            c0=0 # first loop counter
            for imem in tqdm(range(len(lmem))):
                mem=lmem[imem]
                # function that returns directory names where files are located
                [ht2m0, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'his',fo),his,mem,se), 'rb'))
                if fut=='2000-2014':
                    [ht2m1, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'his',fo),fut,mem,se), 'rb'))
                else:
                    [ht2m1, gr] = pickle.load(open('%s/ht2m_%s.%s.%s.pickle' % (mdir(se,'fut',fo),fut,mem,se), 'rb'))

                dt2m0=ht2m0[ixpc,...]-ht2m0[imed,...] # take difference from median in historical
                dt2m1=ht2m1[ixpc,...]-ht2m1[imed,...] # take difference from median in future
                cct2m=dt2m1/dt2m0 # take ratio of distribution widths in future to historical

                # store data
                if c0 == 0:
                    edt2m0=np.empty([len(lmem),len(gr['lat']),len(gr['lon'])])
                    edt2m1=np.empty_like(edt2m0)
                    ecct2m=np.empty_like(edt2m0)
                    c0 = 1

                edt2m0[imem,...]=dt2m0
                edt2m1[imem,...]=dt2m1
                ecct2m[imem,...]=cct2m

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
            # historical
            for ilo in tqdm(range(len(gr['lon']))):
                for ila in range(len(gr['lat'])):
                    avgt2m[ila,ilo] = np.mean(      edt2m0[:,ila,ilo],axis=0)
                    stdt2m[ila,ilo] = np.std(       edt2m0[:,ila,ilo],axis=0)
                    p50t2m[ila,ilo] = np.percentile(edt2m0[:,ila,ilo],50,axis=0)
                    p25t2m[ila,ilo] = np.percentile(edt2m0[:,ila,ilo],25,axis=0)
                    p75t2m[ila,ilo] = np.percentile(edt2m0[:,ila,ilo],75,axis=0)
                    mint2m[ila,ilo] = np.amin(      edt2m0[:,ila,ilo],axis=0)
                    maxt2m[ila,ilo] = np.amax(      edt2m0[:,ila,ilo],axis=0)
                    ptpt2m[ila,ilo] = np.ptp(       edt2m0[:,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgt2m
            stats['stdev'] = stdt2m
            stats['median'] = p50t2m
            stats['prc25'] = p25t2m
            stats['prc75'] = p75t2m
            stats['min'] = mint2m
            stats['max'] = maxt2m
            stats['range'] = ptpt2m

            pickle.dump([stats, gr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (hdir,varn,xpc,his,se), 'wb'), protocol=5)	

            # future
            for ilo in tqdm(range(len(gr['lon']))):
                for ila in range(len(gr['lat'])):
                    avgt2m[ila,ilo] = np.mean(      edt2m1[:,ila,ilo],axis=0)
                    stdt2m[ila,ilo] = np.std(       edt2m1[:,ila,ilo],axis=0)
                    p50t2m[ila,ilo] = np.percentile(edt2m1[:,ila,ilo],50,axis=0)
                    p25t2m[ila,ilo] = np.percentile(edt2m1[:,ila,ilo],25,axis=0)
                    p75t2m[ila,ilo] = np.percentile(edt2m1[:,ila,ilo],75,axis=0)
                    mint2m[ila,ilo] = np.amin(      edt2m1[:,ila,ilo],axis=0)
                    maxt2m[ila,ilo] = np.amax(      edt2m1[:,ila,ilo],axis=0)
                    ptpt2m[ila,ilo] = np.ptp(       edt2m1[:,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgt2m
            stats['stdev'] = stdt2m
            stats['median'] = p50t2m
            stats['prc25'] = p25t2m
            stats['prc75'] = p75t2m
            stats['min'] = mint2m
            stats['max'] = maxt2m
            stats['range'] = ptpt2m

            pickle.dump([stats, gr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (fdir,varn,xpc,fut,se), 'wb'), protocol=5)	


            # ratio (future/historical)
            for ilo in tqdm(range(len(gr['lon']))):
                for ila in range(len(gr['lat'])):
                    avgt2m[ila,ilo] = np.mean(      ecct2m[:,ila,ilo],axis=0)
                    stdt2m[ila,ilo] = np.std(       ecct2m[:,ila,ilo],axis=0)
                    p50t2m[ila,ilo] = np.percentile(ecct2m[:,ila,ilo],50,axis=0)
                    p25t2m[ila,ilo] = np.percentile(ecct2m[:,ila,ilo],25,axis=0)
                    p75t2m[ila,ilo] = np.percentile(ecct2m[:,ila,ilo],75,axis=0)
                    mint2m[ila,ilo] = np.amin(      ecct2m[:,ila,ilo],axis=0)
                    maxt2m[ila,ilo] = np.amax(      ecct2m[:,ila,ilo],axis=0)
                    ptpt2m[ila,ilo] = np.ptp(       ecct2m[:,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgt2m
            stats['stdev'] = stdt2m
            stats['median'] = p50t2m
            stats['prc25'] = p25t2m
            stats['prc75'] = p75t2m
            stats['min'] = mint2m
            stats['max'] = maxt2m
            stats['range'] = ptpt2m

            pickle.dump([stats, gr], open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (odir,varn,xpc,his,fut,se), 'wb'), protocol=5)	

