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

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

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

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        idirm='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)

        # load budyko
        if cl=='his':
            fn = '%s/%s_%s.%s.pickle' % (idirm,varn,yr,se)
            [ef,arr2,stdev] = pickle.load(open(fn,'rb'))
        elif cl=='fut':
            idir0='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical','mmm',varn)
            fn0 = '%s/%s_%s.%s.pickle' % (idir0,varn,'1980-2000',se)
            [ef0,arr2,_] = pickle.load(open(fn0,'rb'))
            # load change in bc
            idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn)
            idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            [ef1,_,_]=pickle.load(open('%s/%s_%s.%s.pickle' % (idir1,varn,'1980-2000',se), 'rb'))
            [ef2,_,_]=pickle.load(open('%s/%s_%s.%s.pickle' % (idir2,varn,yr,se), 'rb'))
            delef=ef2-ef1
            ef=ef0+delef

        # >95th and mean sm data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mmm',varn2)
        l = pickle.load(open('%s/pcm%s_%s.%s.pickle' % (idir,varn2,yr,se), 'rb'))

        pef=np.empty_like(l)
        for ilo in tqdm(range(ef.shape[2])):
            for ila in range(ef.shape[1]):
                fint=interp1d(arr2,ef[:,ila,ilo],fill_value='extrapolate')
                pef[:,ila,ilo]=fint(l[:,ila,ilo])

        pickle.dump(pef, open('%s/pc.devdbc%s_%s.%s.pickle' % (odir,varn1,yr,se), 'wb'), protocol=5)	
