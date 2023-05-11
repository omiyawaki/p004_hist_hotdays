import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods

# index for location to make plot
iloc=[110,85] # SEA

varn1='ef' # var
varn2='mrsos' # var
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
yr0='2000-2022' # hydroclimate regime years
his='1980-2000'
fut='2080-2100'
pc1=0
pc2=95
pc='%02d+%02d'%(pc1,pc2)
lmd=mods(fo)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for se in lse:
    fig,ax=plt.subplots(figsize=(6,4))
    ax.axhline(0,color='k',linewidth=0.5)

    idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
    # load hydroclimate regime info
    [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
    hr=hr*lm

    xy=np.empty([2,len(lmd)])
    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

        [vn1pc1, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,pc1,his,fut,se), 'rb'))
        [vn1pc2, _] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,pc2,his,fut,se), 'rb'))
        [vn2pc1, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir2,pc1,his,fut,se), 'rb'))
        [vn2pc2, _] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir2,pc2,his,fut,se), 'rb'))

        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        lhr=hr[ila,ilo]
        if np.isnan(lhr):
            continue
        lhr=int(lhr)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s+%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn1,varn2,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        l1pc1=vn1pc1[ila,ilo]
        l1pc2=vn1pc2[ila,ilo]
        l2pc1=vn2pc1[ila,ilo]
        l2pc2=vn2pc2[ila,ilo]
        # xy[0,imd]=l1
        # xy[1,imd]=l2

        # plot scatter of var 1 and var2
        # ax.scatter(l2,l1,s=1,label=md.upper())
        ax.plot(l2pc1,l1pc1,alpha=0,label='%g=%s'%(imd,md.upper()))
        ax.text(l2pc1,l1pc1,imd,ha='center',va='center')
        ax.plot(l2pc2,l1pc2,alpha=0)
        ax.text(l2pc2,l1pc2,imd,color='tab:orange',ha='center',va='center')
        ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
        ax.set_ylabel(r'$EF_\mathrm{%s}-EF_\mathrm{%s}$ (unitless)' % (fut,his))
        ax.set_xlabel(r'$SM_\mathrm{%s}-SM_\mathrm{%s}$ (kg m$^{-2}$)' % (fut,his))
    # m,y0,rv,pv,sterr=linregress(xy[1,:],xy[0,:])
    # xv=np.linspace(xy[1,:].min(),xy[1,:].max(),2)
    # ax.plot(xv,y0+m*xv,color='tab:red')
    # ax.text(0.01,0.99,'$R=%+.2f$'%rv,ha='left',va='top',transform=ax.transAxes)
    plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
    plt.tight_layout()
    plt.savefig('%s/scatter.%s.%s.%+05.1f.%+05.1f.%s.%s.mean+95.pdf' % (odir,varn1,varn2,la,lo,pc,se), format='pdf', dpi=300)
    plt.close()

