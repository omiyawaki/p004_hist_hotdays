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

realm='atmos'
freq='day'
varn1='hfls' # y axis var
varn2='mrsos'# x axis var
repl1=0 # replacement value if nan
repl2=0 # replacement value if nan
varn='%s+%s'%(varn1,varn2)
fo='historical'
yr='1980-2000'
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

def modlogifunc(x,A,x0,k,x2,k2):
    return A / (1 + np.exp(-k*(x-x0))) / (1 + np.exp(-k2*(x-x2)))


for se in lse:
    # list of models
    lmd=mods(fo)

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]

        idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
        odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        if not os.path.exists(odir):
            os.makedirs(odir)

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn1].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2].load()
        gr={}
        gr['lat']=ds1.lat
        gr['lon']=ds1.lon

        # create list to store kdes

        ila=iloc[0]
        ilo=iloc[1]
        l1=vn1[:,ila,ilo]/1000 # rescale to make x and y
        l2=vn2[:,ila,ilo]/1000 # closer to order 1
        a=np.nanmean(l1)
        r1=np.nanmax(l1)-np.nanmin(l1)
        r2=np.nanmax(l2)-np.nanmin(l2)
        k=r1/r2*100
        x1=np.nanmin(l2)
        x2=np.nanmax(l2)
        l1[np.isnan(l1)]=repl1
        l2[np.isnan(l2)]=repl2
        try:
            popt, pcov = curve_fit(modlogifunc, l2, l1, p0=[a,x1,k/2,x2,-k])
        except:
            print('Fit did not converge.')

        pickle.dump(popt, open('%s/rmlogi%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,ila,ilo,se), 'wb'), protocol=5)	
