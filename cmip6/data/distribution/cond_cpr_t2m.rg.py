import os
import sys
sys.path.append('../')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from cmip6util import mods
from glade_utils import grid

# this script creates a histogram of daily temperature for a given year
# at each gridir point. 

realm='atmos' # data realm e.g., atmos, ocean, seaIce, etc
freq='day' # data frequency e.g., day, mon

lnt=[30]
varn0='cpr' # variable name

# lfo = ['historical'] # forcing (e.g., ssp245)
# lcl=['his']
# byr='1980-2000' # output year bounds

lfo = ['ssp370'] # forcing (e.g., ssp245)
lcl = ['fut'] # climatology (fut=future [2030-2050], his=historical [1920-1940])
byr='2080-2100'

# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['jja','djf','son','mam'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# lcl = ['fut','his'] # climatology (fut=future [2030-2050], his=historical [1920-1940])

# percentiles to compute (follows Byrne [2021])
pc = [0,95,99] 

for nt in lnt:
    varn='%s%03d'%(varn0,nt)
    for se in lse:
        for fo in lfo:
            for cl in lcl:
                # list of models
                lmd=mods(fo)

                # years and simulation names
                for imd in tqdm(range(len(lmd))):
                    md=lmd[imd]

                    idirv='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn0)
                    idirt='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'tas')
                    odirt='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'tas')
                    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn0)

                    if not os.path.exists(odir):
                        os.makedirs(odir)

                    c=0 # counter
                    # load temp
                    fn = '%s/cl%s_%s.%s.nc' % (idirt,'tas',byr,se)
                    ds = xr.open_mfdataset(fn)
                    t2m=ds['tas'].load()
                    # load varn
                    fn = '%s/cl%s_%s.%s.nc' % (idirv,varn,byr,se)
                    ds = xr.open_mfdataset(fn)
                    cpr=ds[varn0].load()
                        
                    # save grid info
                    gr = {}
                    gr['lon'] = ds['lon']
                    gr['lat'] = ds['lat']

                    # load percentile values
                    ht2m,_=pickle.load(open('%s/h%s_%s.%s.rg.pickle' % (odirt,'tas',byr,se), 'rb'))	
                    ht2m95=ht2m[-2,...]
                    ht2m99=ht2m[-1,...]

                    # initialize array to store subsampled means data
                    ccpr = np.empty([len(pc), gr['lat'].size, gr['lon'].size])

                    # loop through gridir points to compute percentiles
                    for ln in tqdm(range(gr['lon'].size)):
                        for la in range(gr['lat'].size):
                            lt = t2m[:,la,ln]
                            lv = cpr[:,la,ln]
                            lt95 = ht2m95[la,ln]
                            lt99 = ht2m99[la,ln]
                            ccpr[0,la,ln]=np.nanmean(lv)
                            ccpr[1,la,ln]=np.nanmean(lv[np.where(lt>lt95)])
                            ccpr[2,la,ln]=np.nanmean(lv[np.where(lt>lt99)])

                    pickle.dump([ccpr, gr], open('%s/c%s_%s.%s.rg.pickle' % (odir,varn,byr,se), 'wb'), protocol=5)	
