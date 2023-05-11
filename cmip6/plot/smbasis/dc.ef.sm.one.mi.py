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
lpc=[0,95]
lab=['$\Delta EF$','$EF\Delta P_{SM}$','$P_{SM}\Delta EF$','$\Delta EF \Delta P_{SM}$','Residual']

yr0='2000-2022' # hydroclimate regime years
yr='1980-2000+2080-2100' # hist

fo1='historical'
fo2='ssp370'
fo='%s+%s'%(fo1,fo2)
cl='%s+%s'%(cl1,cl2)

lmd=mods(fo1)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for pc in lpc:
    for se in lse:
        idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
        # load hydroclimate regime info
        [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
        hr=hr*lm
        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        dv=np.empty(len(lmd))
        dc1=np.empty(len(lmd))
        dc2=np.empty(len(lmd))
        dc3=np.empty(len(lmd))
        res=np.empty(len(lmd))
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)

            # dc data
            idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

            fn='%s/dc%s_%s.%g.%g.%s.%02d.pickle' % (idir,varn,yr,ila,ilo,se,pc)
            [idv,idc1,idc2,idc3,ires] = pickle.load(open(fn,'rb'))
            dv[imd]=idv
            dc1[imd]=idc1
            dc2[imd]=idc2
            dc3[imd]=idc3
            res[imd]=ires

        # box plot
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.boxplot([dv,dc1,dc2,dc3,res],labels=lab)
        ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
        ax.set_ylabel(r'$\Delta EF_{SM}$ (unitless)')
        plt.tight_layout()
        plt.savefig('%s/dc.box.%s.%+05.1f.%+05.1f.%s.%02d.pdf' % (odir,varn,la,lo,se,pc), format='pdf', dpi=300)
        plt.close()

        # violin plot
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.violinplot([dv,dc1,dc2,dc3,res])
        ax.set_xticks([1,2,3,4,5])
        ax.set_xticklabels(lab)
        ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
        ax.set_ylabel(r'$\Delta EF_{SM}$ (unitless)')
        plt.tight_layout()
        plt.savefig('%s/dc.violin.%s.%+05.1f.%+05.1f.%s.%02d.pdf' % (odir,varn,la,lo,se,pc), format='pdf', dpi=300)
        plt.close()

