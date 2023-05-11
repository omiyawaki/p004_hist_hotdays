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

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn1='rsfc' # y axis var
varn2='dmrsos'# x axis var
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

        fn = '%s/b%s_%s.%g.%g.%s.dev.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        [irsfc,iarr2] = pickle.load(open(fn,'rb'))

        if imd==0:
            ersfc=np.empty([len(lmd),len(irsfc)])

        ersfc[imd,:]=irsfc

        rsfc=np.nanmean(ersfc,axis=0)

    pickle.dump([rsfc,iarr2,ersfc], open('%s/b%s_%s.%g.%g.%s.dev.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
