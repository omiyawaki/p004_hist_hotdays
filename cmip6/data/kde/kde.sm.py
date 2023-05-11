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

realm='atmos'
freq='day'
varn='mrsos'# 
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
# fo='historical'
# yr='1980-2000'
fo='ssp370'
yr='2080-2100'

for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        fn = '%s/cl%s_%s.%s.nc' % (idir,varn,yr,se)

        ds=xr.open_dataset(fn)
        vn=ds[varn].load()
        gr={}
        gr['lat']=ds.lat
        gr['lon']=ds.lon

        # create list to store kdes
        kde = [ ([0] * len(gr['lon'])) for ila in range(len(gr['lat'])) ]
        for ilo in tqdm(range(len(gr['lon']))):
            for ila in range(len(gr['lat'])):
                l=vn[:,ila,ilo]
                l=l[~np.isnan(l)]
                try:
                    kde[ila][ilo]=gaussian_kde(l)
                except:
                    pass

        pickle.dump(kde, open('%s/k%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
