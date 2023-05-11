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

# fo='historical'
# yr='1980-2000'

fo='ssp370'
yr='2080-2100'

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
        if cl=='his':
            fn = '%s/b%s_%s.%s.dev.pickle' % (idirm,varn,yr,se)
            [ef,arr2] = pickle.load(open(fn,'rb'))
        elif cl=='fut':
            idir0='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical','mmm',varn)
            fn0 = '%s/b%s_%s.%s.dev.pickle' % (idir0,varn,'1980-2000',se)
            [ef0,arr2] = pickle.load(open(fn0,'rb'))
            # load change in bc
            idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn)
            idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            [ef1,_]=pickle.load(open('%s/b%s_%s.%s.dev.pickle' % (idir1,varn,'1980-2000',se), 'rb'))
            [ef2,_]=pickle.load(open('%s/b%s_%s.%s.dev.pickle' % (idir2,varn,yr,se), 'rb'))
            # extrapolate efs
            for ilo in tqdm(range(ef1.shape[2])):
                for ila in range(ef1.shape[1]):
                    lef0=ef0[:,ila,ilo]
                    lef1=ef1[:,ila,ilo]
                    lef2=ef2[:,ila,ilo]
                    if not np.all(np.isnan(lef0)):
                        fint=interp1d(arr2[~np.isnan(lef0)],lef0[~np.isnan(lef0)],bounds_error=False,fill_value='extrapolate')
                        ef0[:,ila,ilo]=fint(arr2)
                    if not np.all(np.isnan(lef1)):
                        fint=interp1d(arr2[~np.isnan(lef1)],lef1[~np.isnan(lef1)],bounds_error=False,fill_value='extrapolate')
                        ef1[:,ila,ilo]=fint(arr2)
                    if not np.all(np.isnan(lef2)):
                        fint=interp1d(arr2[~np.isnan(lef2)],lef2[~np.isnan(lef2)],bounds_error=False,fill_value='extrapolate')
                        ef2[:,ila,ilo]=fint(arr2)
            delef=ef2-ef1
            ef=ef0+delef

        # >95th and mean sm data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn2)
        l = pickle.load(open('%s/pcm%s_%s.%s.dev.pickle' % (idir,varn2,yr,se), 'rb'))

        pef=np.empty_like(l)
        for ilo in tqdm(range(ef.shape[2])):
            for ila in range(ef.shape[1]):
                fint=interp1d(arr2,ef[:,ila,ilo],fill_value='extrapolate')
                pef[:,ila,ilo]=fint(l[:,ila,ilo])

        pickle.dump(pef, open('%s/pc.devdbc%s_%s.%s.dev.pickle' % (odir,varn1,yr,se), 'wb'), protocol=5)	
