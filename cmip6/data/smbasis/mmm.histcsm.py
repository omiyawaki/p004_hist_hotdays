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
varn='mrsos'

for se in lse:
    # list of models
    lmd=mods(fo)

    odir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,'mmm',varn)
    odir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,'mmm',varn)

    if not os.path.exists(odir1):
        os.makedirs(odir1)

    if not os.path.exists(odir2):
        os.makedirs(odir2)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        # >95th and mean sm data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)
        [vn1, _] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir1,varn,yr1,se), 'rb'))
        [vn2, _] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir2,varn,yr2,se), 'rb'))
        idsm=vn2-vn1
        
        if imd==0:
            esm1=np.empty([len(lmd),vn1.shape[0],vn1.shape[1],vn1.shape[2]])
            edsm=np.empty([len(lmd),vn1.shape[0],vn1.shape[1],vn1.shape[2]])

        esm1[imd,:]=vn1
        edsm[imd,:]=idsm

    sm1=np.nanmean(esm1,axis=0)
    sm2=sm1+edsm

    pickle.dump(sm1, open('%s/pc%s_%s.%s.pickle' % (odir1,varn,yr1,se), 'wb'), protocol=5)	
    pickle.dump(sm2, open('%s/pc%s_%s.%s.pickle' % (odir2,varn,yr2,se), 'wb'), protocol=5)	
