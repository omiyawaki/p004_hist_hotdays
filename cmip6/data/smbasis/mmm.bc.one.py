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

for se in lse:
    # list of models
    lmd=mods(fo)

    odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        [ief,iarr2,istdev] = pickle.load(open(fn,'rb'))

        # # scatter sm data
        # idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
        # fn = '%s/cl%s_%s.%s.nc' % (idir,varn2,yr,se)
        # ds=xr.open_dataset(fn)
        # vn=ds[varn2].load()
        # l=vn[:,iloc[0],iloc[1]].data
        # ief[iarr2<np.min(l)]=np.nan
        # ief[iarr2>np.max(l)]=np.nan

        if imd==0:
            eef=np.empty([len(lmd),len(ief)])
            estdev=np.empty([len(lmd),len(ief)])

        eef[imd,:]=ief
        estdev[imd,:]=istdev

        ef=np.nanmean(eef,axis=0)
        stdev=np.sqrt(np.nansum(estdev**2,axis=0))

        # ef=np.mean(eef,axis=0)
        # stdev=np.sqrt(np.sum(estdev**2,axis=0))

    pickle.dump([ef,iarr2,stdev,eef,estdev], open('%s/%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
