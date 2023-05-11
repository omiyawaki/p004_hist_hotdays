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
varn1='ef' # y axis var
varn2='dmrsos'# x axis var
varn='%s+%s'%(varn1,varn2)

for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        idirm='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)

        # load budyko
        fn = '%s/b%s_%s.%g.%g.%s.dev.pickle' % (idirm,varn,yr,iloc[0],iloc[1],se)
        [ef,arr2] = pickle.load(open(fn,'rb'))

        # >95th and mean sm data
        if cl=='his':
            idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn2)
            l = pickle.load(open('%s/pcm%s_%s.%g.%g.%s.dev.pickle' % (idir,varn2,yr,iloc[0],iloc[1],se), 'rb'))
        elif cl=='fut':
            idir0 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical','mmm',varn2)
            l0 = pickle.load(open('%s/pcm%s_%s.%g.%g.%s.dev.pickle' % (idir0,varn2,'1980-2000',iloc[0],iloc[1],se), 'rb'))
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn2)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
            [vn1, _] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir1,varn2,'1980-2000',se), 'rb'))
            l1=vn1[:,iloc[0],iloc[1]]
            [vn2, _] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir2,varn2,yr,se), 'rb'))
            l2=vn2[:,iloc[0],iloc[1]]
            l=l0+l2-l1
            # l[0]=0 # by definition; in practice small variations arise due to numerical error propagating

        fint=interp1d(arr2,ef)
        pef=fint(l)

        pickle.dump(pef, open('%s/pc.devcsm%s_%s.%g.%g.%s.dev.pickle' % (odir,varn1,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
