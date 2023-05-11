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
from scipy.optimize import curve_fit
from regions import rinfo
from glade_utils import grid
from cmip6util import mods,emem,simu,year

# this script aggregates the histogram of daily temperature for a given region on interest

# index of select location
# iloc=[110,85] # SEA
iloc=[135,200] # SWUS

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn1='hfls' # y axis var
varn2='mrsos'# x axis var
repl1=0 # replacement value if nan
repl2=0 # replacement value if nan
varn='%s+%s'%(varn1,varn2)
# def logifunc(x,A,x0,k,off):
#     return A / (1 + np.exp(-k*(x-x0)))+off

def logifunc(x,A,x0,k):
    return A / (1 + np.exp(-k*(x-x0)))

for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # scatter data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

        ds1=xr.open_dataset(fn1)
        l1=ds1[varn1][:,iloc[0],iloc[1]].load().data
        ds2=xr.open_dataset(fn2)
        l2=ds2[varn2][:,iloc[0],iloc[1]].load().data

        # load regression coeffs
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        popt=pickle.load(open('%s/rwgtlogi%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'rb'))	
        a=popt[0]; x0=popt[1]; k=popt[2]
        th=[0,0]
        th[0]=x0-1/k # water limited regime threshold (25)
        th[1]=x0+1/k # water saturated regime threshold (75)
        # th[0]=x0-3/k # water limited regime threshold (5)
        # th[1]=x0+3/k # water saturated regime threshold (95)

        # fraction of days in each regime
        nd=len(l2)
        fr=[0,0,0]
        if k>0:
            fr[0]=len(l2[l2<=th[0]])/nd
            fr[1]=len(l2[np.logical_and(l2>th[0],l2<=th[1])])/nd
            fr[2]=len(l2[l2>th[1]])/nd
        else:
            fr[2]=1

        pickle.dump([th,fr], open('%s/bcr%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
