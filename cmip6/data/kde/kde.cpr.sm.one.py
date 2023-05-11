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

# index for location to make plot
iloc=[110,85] # SEA
# iloc=[135,200] # SWUS

# lnt=[30,60,90]
lnt=[7,14]

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

lse = ['jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn0='cpr' # y axis var
varn2='mrsos'# x axis var

for nt in lnt:
    varn1='%s%03d'%(varn0,nt) # y axis var
    varn='%s+%s'%(varn1,varn2)
    for se in lse:
        # list of models
        lmd=mods(fo)

        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn0)
            idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
            fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

            ds1=xr.open_dataset(fn1)
            vn1=ds1[varn0].load()
            ds2=xr.open_dataset(fn2)
            vn2=ds2[varn2].load()
            gr={}
            gr['lat']=ds1.lat
            gr['lon']=ds1.lon

            # create list to store kdes
            ila=iloc[0]
            ilo=iloc[1]
            l1=vn1[:,ila,ilo]
            l2=vn2[:,ila,ilo]
            # remove all nans
            l1=l1[~np.isnan(l1)]
            l2=l2[~np.isnan(l2)]
            try:
                kde=gaussian_kde(np.vstack([l2,l1]))
            except:
                pass

            pickle.dump(kde, open('%s/k%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,ila,ilo,se), 'wb'), protocol=5)	
