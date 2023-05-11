import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from metpy.units import units
from metpy.calc import specific_humidity_from_dewpoint as td2q
from scipy.stats import gaussian_kde
from sfutil import emem,conf,simu,sely
from regions import rinfo

# this script aggregates the histogram of daily temperature for a given region on interest

varn='qt2m'
lfo=['lens']
lre=['sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lcl=['tseries']
byr=[1950,2100]

for re in lre:
    # file where selected region is provided
    rloc,rlat,rlon=rinfo(re)

    for se in lse:
        for fo in lfo:
            for cl in lcl:
                if fo=='lens':
                    idir = '/project/mojave/cesm2/LENS/atm/day_1'
                    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s/%s' % (se,cl,fo,varn)
                else:
                    idir = '/glade/campaign/cesm/collections/CESM2-SF/timeseries/atm/proc/tseries/day_1'
                    odir = '/project/amp/miyawaki/data/p004/hist_hotdays/cesm2-sf/%s/%s/%s/%s' % (se,cl,fo,varn)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                # list of ensemble member numbers
                lmem=emem(fo)
                # years and simulation names
                lyr=sely(fo,cl)

                for imem in tqdm(range(len(lmem))):
                    mem=lmem[imem]
                    for yr in tqdm(lyr):
                        yr0=int(yr[0:4])
                        yr1=int(yr[9:13])
                        cnf=conf(fo,cl,yr=yr0)
                        sim=simu(fo,cl,yr=yr0)
                        if fo=='lens':
                            pf='b.e21.%s.f09_g17.%s.cam.h1'%(cnf,sim[imem])
                        else:
                            pf='b.e21.%s.f09_g17.%s.%s.cam.h1'%(cnf,sim,mem)
                        # load temp
                        fn = '%s/TREFHT/%s.TREFHT.%s.nc' % (idir,pf,yr)
                        ds = xr.open_dataset(fn)
                        t2m = ds['TREFHT'].load()*units.kelvin
                        # load sp humidity
                        fn = '%s/QREFHT/%s.QREFHT.%s.nc' % (idir,pf,yr)
                        ds = xr.open_dataset(fn)
                        q2m = ds['QREFHT'].load()

                        # select data within time of interest if necessary
                        if yr0<byr[0]:
                            yr0=byr[0]
                            t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                            q2m=q2m.sel(time=q2m['time.year']>=byr[0])
                        if yr1>byr[1]:
                            yr1=byr[1]
                            t2m=t2m.sel(time=t2m['time.year']<=byr[1])
                            q2m=q2m.sel(time=q2m['time.year']<=byr[1])

                        # save data year by year
                        for eyr in tqdm(np.arange(yr0,yr1+1)):
                            # select data within time of interest
                            yt2m=t2m.sel(time=t2m['time.year']==eyr)
                            yq2m=q2m.sel(time=q2m['time.year']==eyr)

                            if se != 'ann':
                                yt2m=yt2m.sel(time=yt2m['time.season']==se.upper())
                                yq2m=yq2m.sel(time=yq2m['time.season']==se.upper())
                            gr = {}
                            gr['lon'] = ds['lon']
                            gr['lat'] = ds['lat']

                            # if np.logical_or(np.not_equal(gr['lat'],rlat).any(),np.not_equal(gr['lon'],rlon).any()):
                            if np.logical_or(len(gr['lat'])!=len(rlat),len(gr['lon'])!=len(rlon)):
                                error('Check that selected region is in same grid as data being applied to.')

                            rt2m=np.empty([yt2m.shape[0],len(rloc[0])])
                            rq2m=np.empty([yq2m.shape[0],len(rloc[0])])
                            for it in range(yt2m.shape[0]):
                                lt=yt2m[it,...].data
                                rt2m[it,:]=lt[rloc] # regionally selected data
                                lt=yq2m[it,...].data
                                rq2m[it,:]=lt[rloc] # regionally selected data
                            rt2m=rt2m.flatten()
                            rq2m=rq2m.flatten()
                            kqt2m=gaussian_kde(np.vstack([rt2m,rq2m]))

                            pickle.dump(kqt2m, open('%s/k%s_%g.%s.%s.%s.pickle' % (odir,varn,eyr,mem,re,se), 'wb'), protocol=5)	
