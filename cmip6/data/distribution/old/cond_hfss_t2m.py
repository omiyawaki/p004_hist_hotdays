import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from cmip6util import mods,emem,simu,year
from glade_utils import grid

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon
varn='hfss' # variable name

lfo = ['ssp370'] # forcing (e.g., ssp245)
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
# lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr_his=[1980,2000] # output year bounds
byr_fut=[2080,2100]

# percentiles to compute (follows Byrne [2021])
pc = [0,95,99] 

for se in lse:
    for fo in lfo:
        for cl in lcl:
            # list of models
            lmd=mods(fo)

            # years and simulation names
            if cl == 'fut':
                byr=byr_fut
            elif cl == 'his':
                byr=byr_his

            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                ens=emem(md)
                sim=simu(fo,cl,None)
                grd=grid(fo,cl,md)
                # lyr=year(cl,md,byr)

                idirv='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (sim,freq,varn,md,ens,grd)
                idirt='/project/amp/miyawaki/temp/cmip6/%s/%s/%s/%s/%s/%s' % (sim,freq,'tas',md,ens,grd)
                odirt='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'tas')
                odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                c=0 # counter
                # load temp
                fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idirt,'tas',freq,md,sim,ens,grd)
                ds = xr.open_mfdataset(fn)
                t2m=ds['tas'].load()
                # load varn
                fn = '%s/%s_%s_%s_%s_%s_%s_*.nc' % (idirv,varn,freq,md,sim,ens,grd)
                ds = xr.open_mfdataset(fn)
                sh=ds[varn].load()
                    
                # select data within time of interest
                t2m=t2m.sel(time=t2m['time.year']>=byr[0])
                t2m=t2m.sel(time=t2m['time.year']<=byr[1])
                sh=sh.sel(time=sh['time.year']>=byr[0])
                sh=sh.sel(time=sh['time.year']<=byr[1])

                # select seasonal data if applicable
                if se != 'ann':
                    t2m=t2m.sel(time=t2m['time.season']==se.upper())
                    sh=sh.sel(time=sh['time.season']==se.upper())
                
                # save grid info
                gr = {}
                gr['lon'] = ds['lon']
                gr['lat'] = ds['lat']

                # load percentile values
                ht2m,_=pickle.load(open('%s/h%s_%g-%g.%s.pickle' % (odirt,'tas',byr[0],byr[1],se), 'rb'))	
                ht2m95=ht2m[-2,...]
                ht2m99=ht2m[-1,...]

                # initialize array to store subsampled means data
                csh = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in range(gr['lat'].size):
                        lt = t2m[:,la,ln]
                        lv = sh[:,la,ln]
                        lt95 = ht2m95[la,ln]
                        lt99 = ht2m99[la,ln]
                        csh[0,la,ln]=np.nanmean(lv)
                        csh[1,la,ln]=np.nanmean(lv[np.where(lt>lt95)])
                        csh[2,la,ln]=np.nanmean(lv[np.where(lt>lt99)])

                pickle.dump([csh, gr], open('%s/c%s_%g-%g.%s.pickle' % (odir,varn,byr[0],byr[1],se), 'wb'), protocol=5)	
