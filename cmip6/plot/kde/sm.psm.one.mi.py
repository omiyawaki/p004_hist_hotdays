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

psm=np.arange(100)+1

varn1='mrsos'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

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

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # plot scatter of var 1 and var2
    fig,ax=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
    ax=ax.flatten()
    fig.suptitle(r'%s [%+05.1f,%+05.1f] (%s)' % (se.upper(),la,lo,yr))

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        # binned data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        bef,gr = pickle.load(open('%s/b%s_%s.%s.pickle' % (idir,varn,yr,se), 'rb'))
        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        bef=bef[:,ila,ilo]

        ax[imd].axhline(0,color='k',linewidth=0.5)
        ax[imd].plot(psm,bef,'.r',markersize=2)
        ax[imd].set_xlim([psm[0],psm[-1]])
        ax[imd].set_ylim([0,50])
        ax[imd].set_title(r'%s' % (md.upper()))
        ax[imd].set_xlabel(r'$SM$ Percentile')
        ax[imd].set_ylabel(r'$SM$ (kg m$^{-2}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/scatter.%s.pct.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=128)
    plt.close()

