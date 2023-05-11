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

varn00='ef'
varn0='mrsos'
varn='ef+mrsos'
lse = ['jja'] # season (ann, djf, mam, jja, son)
pc1=0 # 0= mean
pc2=1 # 1= >95th

yr0='2000-2022' # hydroclimate regime years
yr1='1980-2000' # hist
yr2='2080-2100' # fut

fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
cl='%s+%s'%(cl1,cl2)

lmd=mods(fo1)

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

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # plot scatter of var 1 and var2
    fig,ax=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
    ax=ax.flatten()
    fig.suptitle(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        # pdf data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

        fn1 = '%s/%s_%s.%g.%g.%s.pickle' % (idir1,varn,yr1,iloc[0],iloc[1],se)
        fn2 = '%s/%s_%s.%g.%g.%s.pickle' % (idir2,varn,yr2,iloc[0],iloc[1],se)
        [ef1,sm1,sd1] = pickle.load(open(fn1,'rb'))
        [ef2,sm2,sd2] = pickle.load(open(fn2,'rb'))

        # >95th and mean data
        idir1c1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn00)
        idir1c2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn00)
        idir2c1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn0)
        idir2c2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn0)
        [vn1c1, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir1c1,varn00,yr1,se), 'rb'))
        [vn1c2, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir1c2,varn00,yr2,se), 'rb'))
        [vn2c1, gr] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir2c1,varn0,yr1,se), 'rb'))
        [vn2c2, gr] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir2c2,varn0,yr2,se), 'rb'))
        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        # his mean
        l1c1pc1=vn1c1[pc1,ila,ilo]
        l2c1pc1=vn2c1[pc1,ila,ilo]
        # his 95th
        l1c1pc2=vn1c1[pc2,ila,ilo]
        l2c1pc2=vn2c1[pc2,ila,ilo]
        # fut mean
        l1c2pc1=vn1c2[pc1,ila,ilo]
        l2c2pc1=vn2c2[pc1,ila,ilo]
        # fut 95th
        l1c2pc2=vn1c2[pc2,ila,ilo]
        l2c2pc2=vn2c2[pc2,ila,ilo]

        # scatter sm data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn0)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn0)
        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn0,yr1,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn0,yr2,se)
        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn0].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn0].load()
        gr={}
        gr['lat']=ds1.lat
        gr['lon']=ds1.lon
        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        l1=vn1[:,ila,ilo].data
        l2=vn2[:,ila,ilo].data
        sm1[sm1<np.min(l1)]=np.nan
        sm1[sm1>np.max(l1)]=np.nan
        sm2[sm2<np.min(l2)]=np.nan
        sm2[sm2>np.max(l2)]=np.nan

        ax[imd].axhline(0,color='k',linewidth=0.5)
        ax[imd].fill_between(sm1,ef1-sd1,ef1+sd1,color='tab:blue',ec=None,alpha=0.3)
        ax[imd].fill_between(sm2,ef2-sd2,ef2+sd2,color='tab:orange',ec=None,alpha=0.3)
        ax[imd].plot(sm1,ef1,color='tab:blue',label='1980-2000')
        ax[imd].plot(sm2,ef2,color='tab:orange',label='2080-2100')
        ax[imd].plot(l2c1pc1,l1c1pc1,'.',markersize=8,color='tab:blue')
        ax[imd].plot(l2c2pc1,l1c2pc1,'.',markersize=8,color='tab:orange')
        ax[imd].plot(l2c1pc2,l1c1pc2,'+',markersize=8,color='tab:blue')
        ax[imd].plot(l2c2pc2,l1c2pc2,'+',markersize=8,color='tab:orange')
        ax[imd].set_title(r'%s' % (md.upper()))
        ax[imd].set_xlim([0,50])
        ax[imd].set_ylim([0,1.5])
        ax[imd].set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax[imd].set_ylabel(r'$EF_{SM}$ (unitless)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=128)
    plt.close()

