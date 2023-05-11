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

def logifunc(x,A,x0,k,off):
    return A / (1 + np.exp(-k*(x-x0)))+off

# def logifunc(x,A,x0,k):
#     return A / (1 + np.exp(-k*(x-x0)))

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
        vn1=ds1[varn1][:,iloc[0],iloc[1]].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2][:,iloc[0],iloc[1]].load()


        # kde eval params
        xbnd=[np.nanmin(vn2),np.nanmax(vn2)]
        ybnd=[np.nanmin(vn1),np.nanmax(vn1)]
        msm,mlh=np.mgrid[xbnd[0]:xbnd[1]:100j,ybnd[0]:ybnd[1]:100j]
        abm=np.vstack([msm.ravel(),mlh.ravel()])

        # kde data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/k%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        kde = pickle.load(open(fn,'rb'))
        pdf=np.reshape(kde(abm).T,msm.shape)

        # locate median of KDE
        cdf=np.cumsum(pdf,axis=1)
        print(cdf[0,:])
        med=np.nanpercentile(cdf,50,axis=1,method='closest_observation')
        print(np.argwhere(cdf==med))
        sys.exit()
        print(np.equal(cdf,med)[0,:])
        imd=np.where(cdf==med)[0]
        print(mlh[0,imd].shape)
        sys.exit()

        # locate mode of KDE
        # imax=np.argmax(pdf,axis=1)
        # lhmax=mlh[0,imax]

        a=np.nanmax(lhmax)
        im=np.argmax(pdf)
        mx=msm.flatten()[im]
        my=mlh.flatten()[im]
        x0=mx
        ixmy=np.where(lhmax==my)[0][0]
        k=(lhmax[ixmy+5]-lhmax[ixmy-5])/(msm[ixmy+5,0]-msm[ixmy-5,0])
        try:
            popt, pcov = curve_fit(logifunc, msm[:,0], lhmax, p0=[a,x0,k,0])
            # popt, pcov = curve_fit(logifunc, msm[:,0], lhmax, p0=[a,x0,k,0],bounds=([a-np.nanmin(lhmax),-np.inf,-np.inf,0],np.inf))
            # popt, pcov = curve_fit(logifunc, msm[:,0], lhmax, p0=[a,x0,k])
        except:
            print('Fit did not converge.')
            break

        print(popt)

        pickle.dump([msm[:,0],lhmax], open('%s/rkdemax%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
        pickle.dump(popt, open('%s/rkdelogi%s_%s.%g.%g.%s.pickle' % (odir,varn,yr,iloc[0],iloc[1],se), 'wb'), protocol=5)	
