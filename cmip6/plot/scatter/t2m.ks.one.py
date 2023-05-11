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

varn1='tas' # yaxis var
varn2='mrsos' # xaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)
sm=np.linspace(0,100,1000)

fo='ssp370'
cl='fut-his'
yr0='2000-2022' # hydroclimate regime years
his='1980-2000'
fut='2080-2100'
lpc=[95,99]
lmd=mods(fo)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for pc in lpc:
    for se in lse:
        fig,ax=plt.subplots(figsize=(6,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.axvline(0,color='k',linewidth=0.5)

        idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
        # load hydroclimate regime info
        [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
        hr=hr*lm

        xy=np.empty([2,len(lmd)])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
            idir2p1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn2)
            idir2p2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'fut','ssp370',md,varn2)

            [vn1, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir1,varn1,pc,his,fut,se), 'rb'))
            fn1 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir2p1,varn2,his,iloc[0],iloc[1],se)
            fn2 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir2p2,varn2,fut,iloc[0],iloc[1],se)
            kde1 = pickle.load(open(fn1,'rb'))
            kde2 = pickle.load(open(fn2,'rb'))
            pdf1=kde1(sm)
            pdf2=kde2(sm)
            dsm=sm[1]-sm[0]
            cdf1=np.cumsum(pdf1*dsm)
            cdf2=np.cumsum(pdf2*dsm)
            dcdf=cdf2-cdf1
            iks=np.argmax(np.abs(dcdf))
            print(iks)
            ks=dcdf[iks]
            print(ks)

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

            l1=vn1[ila,ilo]
            l2=ks
            xy[0,imd]=l1
            xy[1,imd]=l2

            # plot scatter of var 1 and var2
            # ax.scatter(l2,l1,s=1,label=md.upper())
            ax.plot(l2,l1,alpha=0,label='%g=%s'%(imd,md.upper()))
            ax.text(l2,l1,imd,ha='center',va='center')
            ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
            ax.set_xlabel(r'KS Statistic of $\Delta P_{SM}$')
            ax.set_ylabel(r'$(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}-(T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m})_\mathrm{%s}$ (K)' % (pc,fut,pc,his))
        m,y0,rv,pv,sterr=linregress(xy[1,:],xy[0,:])
        xv=np.linspace(xy[1,:].min(),xy[1,:].max(),2)
        ax.plot(xv,y0+m*xv,color='tab:red')
        ax.text(0.01,0.99,'$R^2=%.2f$'%rv**2,ha='left',va='top',transform=ax.transAxes)
        plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
        plt.tight_layout()
        plt.savefig('%s/scatter.%s.ks%s.%+05.1f.%+05.1f.%s.%s.pdf' % (odir,varn1,varn2,la,lo,pc,se), format='pdf', dpi=300)
        plt.close()
