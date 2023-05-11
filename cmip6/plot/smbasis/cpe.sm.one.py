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

        for md in tqdm(lmd):
            print(md)

            # kde data
            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
            idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

            fn1 = '%s/%s_%s.%g.%g.%s.pickle' % (idir1,varn,yr1,iloc[0],iloc[1],se)
            fn2 = '%s/%s_%s.%g.%g.%s.pickle' % (idir2,varn,yr2,iloc[0],iloc[1],se)
            [cpe1,sm1,sd1] = pickle.load(open(fn1,'rb'))
            [cpe2,sm2,sd2] = pickle.load(open(fn2,'rb'))
            # convert to mm/d
            cpe1=cpe1*86400
            cpe2=cpe2*86400
            sd1=sd1*86400
            sd2=sd2*86400

            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # plot pdfs of var (hist and fut)
            fig,ax=plt.subplots(figsize=(5,4))
            ax.axhline(0,color='k',linewidth=0.5)
            # ax.fill_between(sm1,cpe1-sd1,cpe1+sd1,color='tab:blue',ec=None,alpha=0.3)
            # ax.fill_between(sm2,cpe2-sd2,cpe2+sd2,color='tab:orange',ec=None,alpha=0.3)
            ax.plot(sm1,cpe1,color='tab:blue',label='1980-2000')
            ax.plot(sm2,cpe2,color='tab:orange',label='2080-2100')
            ax.set_xlim([0,50])
            ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
            ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
            ax.set_ylabel(r'PDF of %g-day $(P-E)_{SM}$'%nt)
            plt.legend(frameon=False,loc='upper left')
            plt.tight_layout()
            plt.savefig('%s/%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
            plt.close()

