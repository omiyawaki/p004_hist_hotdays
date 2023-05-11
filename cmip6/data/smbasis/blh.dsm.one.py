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
from scipy.ndimage import generic_filter
from regions import rinfo
from glade_utils import grid
from cmip6util import mods,emem,simu,year

# this script aggregates the histogram of daily temperature for a given region on interest

n_smooth=5

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
varn1='hfls' # y axis var
varn2='dmrsos'# x axis var
vsm=np.linspace(-30,30,100)
varn='%s+%s'%(varn1,varn2)

for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/b%s_%s.%s.pickle' % (idir,varn,yr,se)
        bhfls,_ = pickle.load(open(fn,'rb'))
        bhfls=bhfls[:,iloc[0],iloc[1]]
        # load sm
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'dmrsos+dmrsos')
        fn = '%s/b%s_%s.%s.pickle' % (idir,'dmrsos+dmrsos',yr,se)
        bsm,_ = pickle.load(open(fn,'rb'))
        bsm=bsm[:,iloc[0],iloc[1]]
        
        # interpolate
        fint=interp1d(bsm,bhfls,kind='linear',bounds_error=False,fill_value=np.nan)
        hfls=fint(vsm)

        # smooth data with running mean
        hfls=generic_filter(hfls,np.nanmean,size=n_smooth,mode='constant',cval=np.nan) 

        if not os.path.exists(odir):
            os.makedirs(odir)

        ila=iloc[0]
        ilo=iloc[1]
        pickle.dump([hfls,vsm], open('%s/b%s_%s.%g.%g.%s.dev.pickle' % (odir,varn,yr,ila,ilo,se), 'wb'), protocol=5)	
