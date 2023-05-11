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

varn='ef+mrsos'
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
        delef=ef2-ef1
        delsd=np.sqrt(sd1**2+sd2**2)

        ax[imd].axhline(0,color='k',linewidth=0.5)
        ax[imd].fill_between(sm1,delef-delsd,ef1+sd1,color='k',ec=None,alpha=0.3)
        ax[imd].plot(sm1,delef,color='k')
        ax[imd].set_title(r'%s' % (md.upper()))
        ax[imd].set_xlim([0,50])
        ax[imd].set_ylim([-0.5,0.5])
        ax[imd].set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax[imd].set_ylabel(r'$\Delta EF_{SM}$ (unitless)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/dkde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=128)
    plt.close()

