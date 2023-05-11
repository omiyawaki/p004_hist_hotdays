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
ila=iloc[0]
ilo=iloc[1]

varn1='ef' # var
varn2='ef' # var
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
yr0='2000-2022' # hydroclimate regime years
his='1980-2000'
fut='2080-2100'
lpc=[0,95,99]
lmd=mods(fo)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for pc in lpc:
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
            idir2c1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn2)
            idir2c2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'fut','ssp370',md,varn2)

            [_, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,pc,his,fut,se), 'rb'))
            vn1c1 = pickle.load(open('%s/pc%s_%s.%g.%g.%s.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn1c2 = pickle.load(open('%s/pc%s_%s.%g.%g.%s.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))
            vn2c1 = pickle.load(open('%s/pc%s_%s.%g.%g.%s.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn2c2 = pickle.load(open('%s/pc.mmmcsm%s_%s.%g.%g.%s.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))

            la=gr['lat'][ila]
            lo=gr['lon'][ilo]

            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s+%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn1,varn2,la,lo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            if pc==0:
                l1=vn1c2[0]-vn1c1[0]
                l2=vn2c2[0]-vn2c1[0]
            elif pc==95:
                l1=vn1c2[1]-vn1c1[1]
                l2=vn2c2[1]-vn2c1[1]
            elif pc==99:
                l1=vn1c2[2]-vn1c1[2]
                l2=vn2c2[2]-vn2c1[2]
            xy[0,imd]=l1
            xy[1,imd]=l2

            # plot scatter of var 1 and var2
            # ax.scatter(l2,l1,s=1,label=md.upper())
            # 1:1 line
            ax.plot(l2,l1,alpha=0,label='%g=%s'%(imd,md.upper()))
            ax.text(l2,l1,imd,ha='center',va='center')
            ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
            if pc==0:
                ax.set_ylabel(r'$BC(\overline{SM+\Delta SM})-BC(\overline{SM})$ (unitless)')
                ax.set_xlabel(r'$BC(\overline{SM+\Delta SM_{MMM}})-BC(\overline{SM})$ (unitless)')
            else:
                ax.set_ylabel(r'$BC(SM^{>%s}+\Delta SM^{>%s})-BC(SM^{>%s})$ (unitless)' % (pc,pc,pc))
                ax.set_xlabel(r'$BC(SM^{>%s}+\Delta SM^{>%s}_{MMM})-BC(SM^{>%s})$ (unitless)' % (pc,pc,pc))
        vec=np.linspace(-np.max(np.abs(xy)),np.max(np.abs(xy)),2)
        ax.plot(vec,vec,'--k',linewidth=1)
        m,y0,rv,pv,sterr=linregress(xy[1,:],xy[0,:])
        xv=np.linspace(xy[1,:].min(),xy[1,:].max(),2)
        # ax.plot(xv,y0+m*xv,color='tab:red')
        ax.text(0.01,0.99,'$R=%+.2f$'%rv,ha='left',va='top',transform=ax.transAxes)
        plt.legend(frameon=False,bbox_to_anchor=(1,1),loc='upper left',prop={'size':8})
        plt.tight_layout()
        plt.savefig('%s/scatter.dpc.mmmcsm.%s.%s.%+05.1f.%+05.1f.%s.%s.pdf' % (odir,varn1,varn2,la,lo,pc,se), format='pdf', dpi=300)
        plt.close()

