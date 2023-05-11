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
from scipy.interpolate import interp1d
from regions import rinfo
from glade_utils import grid
from cmip6util import mods,emem,simu,year

# this script aggregates the histogram of daily temperature for a given region on interest

# fo='historical'
# yr='1980-2000'

fo='ssp370'
yr='2080-2100'

lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn1='ef' # y axis var
varn2='mrsos'# x axis var
varn='%s+%s'%(varn1,varn2)

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))
for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn1].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2].load()
        gr={}
        gr['lat']=ds1.lat
        gr['lon']=ds1.lon

        
        # create list to store kdes
        kde = [ ([0] * len(gr['lon'])) for ila in range(len(gr['lat'])) ]
        for ilo in tqdm(range(len(gr['lon']))):
            for ila in range(len(gr['lat'])):
                if np.isnan(lm[ila,ilo]):
                    pass
                    # print('%g deg N %g deg E is an ocean grid, skipping...'%(gr['lat'][ila],gr['lon'][ilo]))
                else:
                    l1=vn1[:,ila,ilo]
                    l2=vn2[:,ila,ilo]
                    # discard unphysical EF data points for SEA location
                    if ila==110 and ilo==85:
                        idxweird=np.where(np.logical_or(l1>100,l1<0))
                        l1[idxweird]=np.nan
                        l2[idxweird]=np.nan
                    # remove all nans
                    l1=l1[~np.isnan(l1)]
                    l2=l2[~np.isnan(l2)]
                    try:
                        kde[ila][ilo]=gaussian_kde(np.vstack([l2,l1]))
                    except:
                        pass

        pickle.dump(kde, open('%s/k%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
