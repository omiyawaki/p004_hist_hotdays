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

def logifunc(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

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
        popt = [ ([0] * len(gr['lon'])) for ila in range(len(gr['lat'])) ]
        for ilo in tqdm(range(len(gr['lon']))):
            for ila in range(len(gr['lat'])):
                l1=vn1[:,ila,ilo]
                l2=vn2[:,ila,ilo]
                r1=np.nanmax(l1)-np.nanmax(l1)
                r2=np.nanmax(l2)-np.nanmax(l2)
                k=r1/r2
                m2=np.nanmean(l2)
                l1[np.isnan(l1)]=repl1
                l2[np.isnan(l2)]=repl2
                try:
                    popt[ila][ilo], pcov = curve_fit(logifunc, l2, l1, p0=[r1,m2,k,0])
                except:
                    print('Fit did not converge.')

        pickle.dump(popt, open('%s/r%s_%s.%s.pickle' % (odir,varn,yr,se), 'wb'), protocol=5)	
