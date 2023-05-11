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

varn='mrsos'
sm=np.linspace(0,100,1000)
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

    for md in tqdm(lmd):
        print(md)

        # kde data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

        fn1 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir1,varn,yr1,iloc[0],iloc[1],se)
        fn2 = '%s/k%s_%s.%g.%g.%s.pickle' % (idir2,varn,yr2,iloc[0],iloc[1],se)
        kde1 = pickle.load(open(fn1,'rb'))
        kde2 = pickle.load(open(fn2,'rb'))
        pdf1=kde1(sm)
        pdf2=kde2(sm)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # plot pdfs of var (hist and fut)
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.plot(sm,pdf1,label='1980-2000')
        ax.plot(sm,pdf2,label='2080-2100')
        ax.set_xlim([0,50])
        ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
        ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax.set_ylabel(r'Probability Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
        plt.close()

