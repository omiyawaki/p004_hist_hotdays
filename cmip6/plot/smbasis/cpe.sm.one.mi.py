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

lnt=[30]
varn1='cpe'
varn2='mrsos'

lse = ['jja'] # season (ann, djf, mam, jja, son)

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

for nt in lnt:
    varn='%s%03d+%s'%(varn1,nt,varn2)
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
            [cpe1,sm1,sd1] = pickle.load(open(fn1,'rb'))
            [cpe2,sm2,sd2] = pickle.load(open(fn2,'rb'))

            # scatter sm data
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn2)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn2)
            fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn2,yr1,se)
            fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr2,se)
            ds1=xr.open_dataset(fn1)
            vn1=ds1[varn2].load()
            ds2=xr.open_dataset(fn2)
            vn2=ds2[varn2].load()
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
            ax[imd].fill_between(sm1,cpe1-sd1,cpe1+sd1,color='tab:blue',ec=None,alpha=0.3)
            ax[imd].fill_between(sm2,cpe2-sd2,cpe2+sd2,color='tab:orange',ec=None,alpha=0.3)
            ax[imd].plot(sm1,cpe1,color='tab:blue',label='1980-2000')
            ax[imd].plot(sm2,cpe2,color='tab:orange',label='2080-2100')
            ax[imd].set_title(r'%s' % (md.upper()))
            ax[imd].set_xlim([0,50])
            ax[imd].set_ylim([-10,30])
            ax[imd].set_xlabel(r'$SM$ (kg m$^{-2}$)')
            ax[imd].set_ylabel(r'%g-day $(P-E)_{SM}$ (mm d$^{-1}$)'%nt)
            plt.legend()
            plt.tight_layout()
            plt.savefig('%s/kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=128)
        plt.close()

