import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from cmip6util import mods

# index for location to make plot
iloc=[110,85] # SEA
# iloc=[135,200] # SWUS

varn1='ef'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
msm,mlh=np.mgrid[0:100:100j,0:250:250j]
abm=np.vstack([msm.ravel(),mlh.ravel()])
lse = ['jja'] # season (ann, djf, mam, jja, son)

# fo='historical'
# yr='1980-2000'

fo='ssp370'
yr='2080-2100'

yr0='2000-2022' # hydroclimate regime years
lmd=mods(fo)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for se in lse:
    idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
    # load hydroclimate regime info
    [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
    hr=hr*lm
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    for md in tqdm(lmd):
        print(md)

        # pdf data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/rpdf%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        pdf = pickle.load(open(fn,'rb'))

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # scatter data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn1].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2].load()
        gr={}
        gr['lat']=ds1.lat
        gr['lon']=ds1.lon

        # [vn1,gr] = pickle.load(open(fn1,'rb'))
        # [vn2,_] = pickle.load(open(fn2,'rb'))

        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        lhr=hr[ila,ilo]
        if np.isnan(lhr):
            continue
        lhr=int(lhr)

        l1=vn1[:,ila,ilo]
        l2=vn2[:,ila,ilo]
        if md=='CMCC-ESM2':
            idxweird=np.where(np.logical_or(l1>100,l1<0))
            l1[idxweird]=np.nan
            l2[idxweird]=np.nan

        # plot scatter of var 1 and var2
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.scatter(l2,l1,c=pdf,s=0.5,label=hrn[lhr])
        ax.set_title(r'%s %s [%+05.1f,%+05.1f] (%s)' % (se.upper(),md.upper(),la,lo,yr))
        ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax.set_ylabel(r'$LH/R_{SFC}$ (unitless)')
        # plt.legend()
        plt.tight_layout()
        plt.savefig('%s/scatter.pdf.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
        plt.close()

