import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cmip6util import mods

varn1='hfls' # yaxis var
varn2='mrsos' # xaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)
skipocn=True # skip ocean grids?

fo='historical'
yr0='2000-2022' # hydroclimate regime years
yr='1980-2000'
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

    for md in tqdm(lmd):
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

        fn1 = '%s/cl%s_%s.%s.pickle' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.pickle' % (idir2,varn2,yr,se)

        [vn1,gr] = pickle.load(open(fn1,'rb'))
        [vn2,_] = pickle.load(open(fn2,'rb'))

        for ilo in tqdm(range(len(gr['lon']))):
            for ila in range(len(gr['lat'])):
                la=gr['lat'][ila]
                lo=gr['lon'][ilo]
                lhr=hr[ila,ilo]
                if np.isnan(lhr):
                    continue
                lhr=int(lhr)

                odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s+%s/gridpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn1,varn2,la,lo)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                l1=vn1[:,ila,ilo]
                l2=vn2[:,ila,ilo]

                # plot scatter of var 1 and var2
                fig,ax=plt.subplots(figsize=(5,4))
                ax.axhline(0,color='k',linewidth=0.5)
                ax.scatter(l2,l1,c=cm[lhr],s=0.5,label=hrn[lhr])
                ax.set_title(r'%s %s [%+05.1f,%+05.1f] (%s)' % (se.upper(),md.upper(),la,lo,yr))
                ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
                ax.set_ylabel(r'$LH$ (W m$^{-2}$)')
                plt.legend()
                plt.tight_layout()
                plt.savefig('%s/scatter.%s.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn1,varn2,la,lo,se), format='pdf', dpi=300)
                plt.close()

