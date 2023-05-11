import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import dask
from dask.diagnostics import ProgressBar
import dask.multiprocessing
import pickle
import pwlf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from tqdm import tqdm
from cmip6util import mods
from utils import corr,corr2d,monname
from regions import pointlocs

# lre=['zambia','amazon','sahara','sea']
lre=['yuma']

nt=7 # window size in days
lpc=np.concatenate((np.arange(0,50,10),np.arange(50,75,5),np.arange(75,95,2.5),np.arange(95,100,1)))
nseg=2 # number of line segments for pwlf
ndeg=1 # degree of polynomial fit
pref1='p'
varn1='hfls'
pref2='p'
varn2='mrsos'
pref3='p'
varn3='pr'
pref4='p'
varn4='tas'
varn='%s%s+%s%s'%(pref1,varn1,pref2,varn2)
se = 'sc'

fo='historical' # forcings 
yr='1980-2000'

fof='ssp370' # forcings 
yrf='2080-2100'

mmm=True

troplat=20    # latitudinal bound of tropics

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

if mmm:
    md='mmm'
else:
    md='mi'
idir1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mi',varn1)
idir2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mi',varn2)
idirf1 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fof,'mi',varn1)
idirf2 = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fof,'mi',varn2)
idirg = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s' % (se,fo,'mmm',varn4)

ri1=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir1,pref1,varn1,yr,se), 'rb'))
ri2=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idir2,pref2,varn2,yr,se), 'rb'))
rif1=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idirf1,pref1,varn1,yrf,se), 'rb'))
rif2=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idirf2,pref2,varn2,yrf,se), 'rb'))
_,_,gr=pickle.load(open('%s/%s%s_%s.%s.pickle' % (idirg,pref4,varn4,yr,se), 'rb'))

# rs1=pickle.load(open('%s/std%s%s_%s.%s.pickle' % (idir1,pref1,varn1,yr,se), 'rb'))
# rs2=pickle.load(open('%s/std%s%s_%s.%s.pickle' % (idir2,pref2,varn2,yr,se), 'rb'))

for re in lre:
    iloc=pointlocs(re)
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    odir = '/project/amp02/miyawaki/data/p004/cmip6/%s/%s/%s/%s/selpoint/%s' % (se,fo,md,varn,re)
    if not os.path.exists(odir):
        os.makedirs(odir)

    i1=ri1[...,iloc[0],iloc[1]]
    i2=ri2[...,iloc[0],iloc[1]]
    if1=rif1[...,iloc[0],iloc[1]]
    if2=rif2[...,iloc[0],iloc[1]]
    id1=if1-i1
    id2=if2-i2

    mmm1=np.nanmean(i1,axis=0)
    mmm2=np.nanmean(i2,axis=0)
    mmmf1=np.nanmean(if1,axis=0)
    mmmf2=np.nanmean(if2,axis=0)
    mmmd1=np.nanmean(id1,axis=0)
    mmmd2=np.nanmean(id2,axis=0)

    pct=np.tile(lpc,(i1.shape[0],i1.shape[1],1))

    oname='%s/pwlf.%gd%g.bc.%s_%s.%s' % (odir,nseg,ndeg,varn,yr,se)

    if mmm:
        i1=mmm1
        i2=mmm2
        if1=mmmf1
        if2=mmmf2
        id1=mmmd1
        id2=mmmd2

        ad1=id1 # actual lh change
        pd1=np.empty_like(i1) # predicted dlh from piecewise BC with actual dSM
        pd1mmm=np.empty_like(i1) # predicted dlh from from piecewise BC with mmm dSM
        for mon in range(i1.shape[0]):
            mi1=i1[mon,...]
            mi2=i2[mon,...]
            mif1=if1[mon,...]
            mif2=if2[mon,...]
            mid1=id1[mon,...]
            mid2=id2[mon,...]
            mpct=pct[0,mon,...]

            # use local maximum of a cubic polynomial fit as breaking point of piecewise linear fit
            pa=np.polynomial.polynomial.polyfit(mi2,mi1,3) # polynomial fit
            xv1=-(2*pa[2]+np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
            xv2=-(2*pa[2]-np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
            cc1=2*pa[2]+6*pa[3]*xv1
            cc2=2*pa[2]+6*pa[3]*xv2
            if cc1<0:
                bp=xv1
            else:
                bp=xv2
            mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
            fp=mp.fit_with_breaks([np.min(mi2),bp,np.max(mi2)])
            pd1[mon,:]=mp.predict(mif2)-mp.predict(mi2)
            pd1mmm[mon,:]=mp.predict(mi2+mmmd2[mon,:])-mp.predict(mi2)

        pickle.dump([ad1,pd1,pd1mmm,pct],open('%s.pickle'%oname,'wb'),protocol=5)

    else:
        ad1=id1 # actual lh change
        pd1=np.empty_like(i1) # predicted dlh from piecewise BC with actual dSM
        pd1mmm=np.empty_like(i1) # predicted dlh from from piecewise BC with mmm dSM
        for md in tqdm(range(i1.shape[0])):
            for mon in range(i1.shape[1]):
                mi1=i1[md,mon,...]
                mi2=i2[md,mon,...]
                mif1=if1[md,mon,...]
                mif2=if2[md,mon,...]
                mid1=id1[md,mon,...]
                mid2=id2[md,mon,...]
                mpct=pct[md,mon,...]

                # use local maximum of a cubic polynomial fit as breaking point of piecewise linear fit
                pa=np.polynomial.polynomial.polyfit(mi2,mi1,3) # polynomial fit
                xv1=-(2*pa[2]+np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
                xv2=-(2*pa[2]-np.sqrt(4*pa[2]**2-12*pa[1]*pa[3]))/(6*pa[3])
                cc1=2*pa[2]+6*pa[3]*xv1
                cc2=2*pa[2]+6*pa[3]*xv2
                if cc1<0:
                    bp=xv1
                else:
                    bp=xv2
                mp=pwlf.PiecewiseLinFit(mi2,mi1,degree=ndeg)
                fp=mp.fit_with_breaks([np.min(mi2),bp,np.max(mi2)])
                pd1[md,mon,:]=mp.predict(mif2)-mp.predict(mi2)
                pd1mmm[md,mon,:]=mp.predict(mi2+mmmd2[mon,:])-mp.predict(mi2)

        pickle.dump([ad1,pd1,pd1mmm,pct],open('%s.pickle'%oname,'wb'),protocol=5)
