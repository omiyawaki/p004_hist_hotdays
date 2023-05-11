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

yr1='1980-2000' # hist
yr2='2080-2100' # fut
yr='%s+%s'%(yr1,yr2)

fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
cl='%s+%s'%(cl1,cl2)

# lpc=[0,95]
lpc=[0,95]
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)

realm='atmos'
freq='day'
varn1='ef' # y axis var
varn2='mrsos'# x axis var
arr1=np.linspace(0,30,1000)
msm,mef=np.mgrid[0:100:1000j,0:30:1000j]
abm=np.vstack([msm.ravel(),mef.ravel()])
varn='%s+%s'%(varn1,varn2)

for pc in lpc:
    for se in lse:
        # list of models
        lmd=mods(fo1)

        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            odir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            # load varn1
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn1)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn1)
            [vn1, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir1,varn1,yr1,se), 'rb'))
            [vn2, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir2,varn1,yr2,se), 'rb'))
            ila=iloc[0]
            ilo=iloc[1]
            if pc==0:
                v1l1=vn1[0,ila,ilo]
                v1l2=vn2[0,ila,ilo]
            elif pc==95:
                v1l1=vn1[1,ila,ilo]
                v1l2=vn2[1,ila,ilo]
            elif pc==99:
                v1l1=vn1[2,ila,ilo]
                v1l2=vn2[2,ila,ilo]
            dv1=v1l2-v1l1

            # load varn2
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn2)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn2)
            [vn1, gr] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir1,varn2,yr1,se), 'rb'))
            [vn2, gr] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir2,varn2,yr2,se), 'rb'))
            ila=iloc[0]
            ilo=iloc[1]
            if pc==0:
                v2l1=vn1[0,ila,ilo]
                v2l2=vn2[0,ila,ilo]
            elif pc==95:
                v2l1=vn1[1,ila,ilo]
                v2l2=vn2[1,ila,ilo]
            elif pc==99:
                v2l1=vn1[2,ila,ilo]
                v2l2=vn2[2,ila,ilo]
            dv2=v2l2-v2l1

            # load ef(sm)
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

            fn1 = '%s/%s_%s.%g.%g.%s.pickle' % (idir1,varn,yr1,iloc[0],iloc[1],se)
            fn2 = '%s/%s_%s.%g.%g.%s.pickle' % (idir2,varn,yr2,iloc[0],iloc[1],se)

            [ef1,sm1,sd1] = pickle.load(open(fn1,'rb'))
            [ef2,sm2,sd2] = pickle.load(open(fn2,'rb'))

            # load pdfs
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn2)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn2)

            fn1 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir1,varn2,yr1,iloc[0],iloc[1],se)
            fn2 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir2,varn2,yr2,iloc[0],iloc[1],se)
            kde1 = pickle.load(open(fn1,'rb'))
            kde2 = pickle.load(open(fn2,'rb'))
            arr2=np.linspace(0,100,1000)
            pdf1=kde1(arr2)
            pdf2=kde2(arr2)

            # compute components
            delef=ef2-ef1
            delpdf=pdf2-pdf1
            dc1=ef1*delpdf
            dc2=delef*pdf1
            dc3=delef*delpdf

            # truncate integration bound if not mean
            if pc != 0:
                print(v2l1)
                dc1=dc1[arr2<v2l1]
                dc2=dc2[arr2<v2l1]
                dc3=dc3[arr2<v2l1]
                arr2=arr2[arr2<v2l1]

            # remove nans and infs
            ndc1=dc1[np.isfinite(dc1)]
            x1=arr2[np.isfinite(dc1)]
            ndc2=dc2[np.isfinite(dc2)]
            x2=arr2[np.isfinite(dc2)]
            ndc3=dc3[np.isfinite(dc3)]
            x3=arr2[np.isfinite(dc3)]
            # ndc1=np.copy(dc1)
            # ndc1[np.isnan(dc1)]=0
            # ndc2=np.copy(dc2)
            # ndc2[np.isnan(dc2)]=0
            # ndc3=np.copy(dc3)
            # ndc3[np.isnan(dc3)]=0

            # integrate
            idc1=np.trapz(ndc1,x=x1)
            idc2=np.trapz(ndc2,x=x2)
            idc3=np.trapz(ndc3,x=x3)
            ires=dv1-idc1-idc2-idc3

            if not os.path.exists(odir):
                os.makedirs(odir)

            pickle.dump([dv1,idc1,idc2,idc3,ires], open('%s/dc%s_%s.%g.%g.%s.%02d.pickle' % (odir,varn,yr,ila,ilo,se,pc), 'wb'), protocol=5)	
