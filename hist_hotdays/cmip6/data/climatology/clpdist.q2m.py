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
lxpc=[95,99] # percentile value from which to take the distance of the median

lfo = ['ssp370'] # forcing (e.g., ssp245)
# lse = ['ann','jja','djf','son','mam'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
cl='fut-his' # climatology (difference between future and historical)
his = '1980-2000' # historical analysis period
fut = '2080-2100' # future analysis period

for xpc in lxpc:
    if xpc==95:
        ixpc=1
    elif xpc==99:
        ixpc=2
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

                [ht2m0, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (mdir(se,'his','historical',md,varn),varn,his,se), 'rb'))
                [ht2m1, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (mdir(se,'fut',fo,md,varn),varn,fut,se), 'rb'))

                dt2m0=ht2m0[ixpc,...]-ht2m0[0,...] # take difference from mean in historical
                dt2m1=ht2m1[ixpc,...]-ht2m1[0,...] # take difference from mean in future
                cct2m=dt2m1/dt2m0 # take ratio of distribution widths in future to historical
                dwt2m=dt2m1-dt2m0 # take difference of distribution widths in future to historical

                # store data
                if c0 == 0:
                    igr={}
                    idt2m0={}
                    idt2m1={}
                    icct2m={}
                    idwt2m={}
                    c0 = 1

                igr[md]=gr
                idt2m0[md]=dt2m0
                idt2m1[md]=dt2m1
                icct2m[md]=cct2m
                idwt2m[md]=dwt2m

                # save climatological percentile distance
                pickle.dump([dt2m0, gr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (hdir,varn,xpc,his,se), 'wb'), protocol=5)	
                pickle.dump([dt2m1, gr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (fdir,varn,xpc,fut,se), 'wb'), protocol=5)	
                # save width ratio
                pickle.dump([cct2m, gr], open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (odir,varn,xpc,his,fut,se), 'wb'), protocol=5)	
                # save width difference
                pickle.dump([dwt2m, gr], open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (odir,varn,xpc,his,fut,se), 'wb'), protocol=5)	

                # save CESM grid for ensemble mean
                if md=='CESM2':
                    egr=gr


            # ENSEMBLE 
            # output directory
            ehdir=mdir(se,'his','historical','mmm',varn)
            efdir=mdir(se,'fut',fo,'mmm',varn)
            edir=mdir(se,cl,fo,'mmm',varn)
            if not os.path.exists(ehdir):
                os.makedirs(ehdir)
            if not os.path.exists(efdir):
                os.makedirs(efdir)
            if not os.path.exists(edir):
                os.makedirs(edir)

            # regrid everything to CESM grid
            edt2m0=np.empty([len(lmd),len(egr['lat']),len(egr['lon'])])
            edt2m1=np.empty([len(lmd),len(egr['lat']),len(egr['lon'])])
            ecct2m=np.empty([len(lmd),len(egr['lat']),len(egr['lon'])])
            edwt2m=np.empty([len(lmd),len(egr['lat']),len(egr['lon'])])
            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                if md!='CESM2-WACCM':
                    fint=interp1d(igr[md]['lat'],idt2m0[md],axis=0,fill_value='extrapolate')
                    lati=fint(egr['lat'])
                    fint=interp1d(igr[md]['lon'],lati,axis=1,fill_value='extrapolate')
                    edt2m0[imd,...]=fint(egr['lon'])

                    fint=interp1d(igr[md]['lat'],idt2m1[md],axis=0,fill_value='extrapolate')
                    lati=fint(egr['lat'])
                    fint=interp1d(igr[md]['lon'],lati,axis=1,fill_value='extrapolate')
                    edt2m1[imd,...]=fint(egr['lon'])

                    fint=interp1d(igr[md]['lat'],icct2m[md],axis=0,fill_value='extrapolate')
                    lati=fint(egr['lat'])
                    fint=interp1d(igr[md]['lon'],lati,axis=1,fill_value='extrapolate')
                    ecct2m[imd,...]=fint(egr['lon'])

                    fint=interp1d(igr[md]['lat'],idwt2m[md],axis=0,fill_value='extrapolate')
                    lati=fint(egr['lat'])
                    fint=interp1d(igr[md]['lon'],lati,axis=1,fill_value='extrapolate')
                    edwt2m[imd,...]=fint(egr['lon'])
                else:
                    edt2m0[imd,...]=idt2m0[md]
                    edt2m1[imd,...]=idt2m1[md]
                    ecct2m[imd,...]=icct2m[md]
                    edwt2m[imd,...]=idwt2m[md]

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
            # historical
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
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

            pickle.dump([stats, egr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (ehdir,varn,xpc,his,se), 'wb'), protocol=5)	

            # future
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
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

            pickle.dump([stats, egr], open('%s/cldist.%s.%02d.%s.%s.pickle' % (efdir,varn,xpc,fut,se), 'wb'), protocol=5)	


            # ratio (future/historical)
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
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

            pickle.dump([stats, egr], open('%s/rdist.%s.%02d.%s.%s.%s.pickle' % (edir,varn,xpc,his,fut,se), 'wb'), protocol=5)	

            # ratio (future/historical)
            for ilo in tqdm(range(len(egr['lon']))):
                for ila in range(len(egr['lat'])):
                    avgt2m[ila,ilo] = np.mean(      edwt2m[:,ila,ilo],axis=0)
                    stdt2m[ila,ilo] = np.std(       edwt2m[:,ila,ilo],axis=0)
                    p50t2m[ila,ilo] = np.percentile(edwt2m[:,ila,ilo],50,axis=0)
                    p25t2m[ila,ilo] = np.percentile(edwt2m[:,ila,ilo],25,axis=0)
                    p75t2m[ila,ilo] = np.percentile(edwt2m[:,ila,ilo],75,axis=0)
                    mint2m[ila,ilo] = np.amin(      edwt2m[:,ila,ilo],axis=0)
                    maxt2m[ila,ilo] = np.amax(      edwt2m[:,ila,ilo],axis=0)
                    ptpt2m[ila,ilo] = np.ptp(       edwt2m[:,ila,ilo],axis=0)

            stats={}
            stats['mean'] = avgt2m
            stats['stdev'] = stdt2m
            stats['median'] = p50t2m
            stats['prc25'] = p25t2m
            stats['prc75'] = p75t2m
            stats['min'] = mint2m
            stats['max'] = maxt2m
            stats['range'] = ptpt2m

            pickle.dump([stats, egr], open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (edir,varn,xpc,his,fut,se), 'wb'), protocol=5)	

