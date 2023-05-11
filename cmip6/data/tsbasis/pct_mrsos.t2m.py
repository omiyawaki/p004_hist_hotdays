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
varn1='tas' # y axis var
varn2='mrsos'# x axis var
varn='%s+%s'%(varn2,varn1)

lfo = ['historical'] # forcing (e.g., ssp245)
lcl = ['his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])

# lfo = ['ssp370'] # forcing (e.g., ssp245)
# lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])

# lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
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

            for imd in tqdm(range(len(lmd))):
                md=lmd[imd]
                ens=emem(md)
                sim=simu(fo,cl,None)
                grd=grid(fo,cl,md)

                # load sm conditioned on tas
                idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                # load tas percentile values
                odirt='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
                ht2m,_=pickle.load(open('%s/h%s_%g-%g.%s.pickle' % (odirt,'tas',byr[0],byr[1],se), 'rb'))	
                ht2m95=ht2m[-2,...]
                ht2m99=ht2m[-1,...]

                # initialize array to store subsampled means data
                cmrsos = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                # loop through gridir points to compute percentiles
                for ln in tqdm(range(gr['lon'].size)):
                    for la in range(gr['lat'].size):
                        lt = t2m[:,la,ln]
                        lv = mrsos[:,la,ln]
                        lt95 = ht2m95[la,ln]
                        lt99 = ht2m99[la,ln]
                        cmrsos[0,la,ln]=np.nanmean(lv)
                        cmrsos[1,la,ln]=np.nanmean(lv[np.where(lt>lt95)])
                        cmrsos[2,la,ln]=np.nanmean(lv[np.where(lt>lt99)])

                pickle.dump([cmrsos, gr], open('%s/p%s_%g-%g.%s.pickle' % (idir,varn2,byr[0],byr[1],se), 'wb'), protocol=5)	
