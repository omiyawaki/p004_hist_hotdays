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
# iloc=[110,85] # SEA
iloc=[135,200] # SWUS

varn1='hfls'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
msm,mlh=np.mgrid[0:100:100j,0:250:250j]
abm=np.vstack([msm.ravel(),mlh.ravel()])
lse = ['jja'] # season (ann, djf, mam, jja, son)

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
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]
    lhr=hr[iloc[0],iloc[1]]
    if np.isnan(lhr):
        continue
    lhr=int(lhr)

    for md in tqdm(lmd):
        print(md)

        # kde data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/k%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        kde = pickle.load(open(fn,'rb'))
        pdf=np.reshape(kde(abm).T,msm.shape)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # max data
        fn = '%s/rkdemax%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        maxsm,maxlh = pickle.load(open(fn,'rb'))

        # plot kde max of var 1 and var2
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.scatter(maxsm,maxlh,c=cm[lhr],s=0.5,label=hrn[lhr])
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.contour(msm,mlh,pdf)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(r'%s %s [%+05.1f,%+05.1f] (%s)' % (se.upper(),md.upper(),la,lo,yr))
        ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax.set_ylabel(r'$LH$ (W m$^{-2}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/scatter.kdemax.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
        plt.close()

