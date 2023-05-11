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

fo1='historical'
yr1='1980-2000'

fo2='ssp370'
yr2='2080-2100'

fo=fo2

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

        idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
        idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

        fn1 = '%s/%s_%s.%s.pickle' % (idir1,varn,yr1,se)
        fn2 = '%s/%s_%s.%s.pickle' % (idir2,varn,yr2,se)
        [ief1,iarr2,_] = pickle.load(open(fn1,'rb'))
        [ief2,_,_] = pickle.load(open(fn2,'rb'))

        if imd==0:
            eef=np.empty([len(lmd),ief1.shape[0],ief1.shape[1],ief1.shape[2]])

        eef[imd,:,:,:]=ief2-ief1

    ef=np.nanmean(eef,axis=0)

    pickle.dump([ef,iarr2], open('%s/d%s_%s.%s.%s.pickle' % (odir,varn,yr1,yr2,se), 'wb'), protocol=5)	
