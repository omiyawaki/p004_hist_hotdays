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

varn1='tas'
varn2='ta500'
varn='%s+%s'%(varn1,varn2) # yaxis var
mta500,mtas=np.mgrid[230:280:50j,270:335:65j]
abm=np.vstack([mta500.ravel(),mtas.ravel()])
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='historical'
yr0='2000-2022' # hydroclimate regime years
yr='1980-2000'
# lmd=mods(fo)
lmd=['CESM2']

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
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/k%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        kde = pickle.load(open(fn,'rb'))
        # pdf=np.reshape(kde(abm).T,mta500.shape)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # scatter data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
        idir3 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'predtm')

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)
        fn3 = '%s/cl%s_%s.%s.nc' % (idir3,'predtm',yr,se)

        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn1].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2].load()
        ds3=xr.open_dataset(fn3)
        vn3=ds3['predtm'].load()
        gr={}
        gr['lat']=ds1.lat
        gr['lon']=ds1.lon

        # [vn1,gr] = pickle.load(open(fn1,'rb'))
        # [vn2,_] = pickle.load(open(fn2,'rb'))

        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        lhr=hr[ila,ilo]
        if np.isnan(lhr):
            continue
        lhr=int(lhr)

        l1=vn1[:,ila,ilo]
        l2=vn2[:,ila,ilo]
        l3=vn3[:,ila,ilo]
        abm=np.vstack([l2,l1])
        pdf=kde(abm)

        # plot scatter of var 1 and var2
        fig,ax=plt.subplots(figsize=(5,4))
        # ax.scatter(l2,l1,c=cm[lhr],s=0.5,label=hrn[lhr])
        ax.scatter(l2,l1,c=pdf,s=0.5)
        ax.scatter(l2,l3,c='k',s=0.5)
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        # ax.contour(mta500,mtas,pdf)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(r'%s %s [%+05.1f,%+05.1f] (%s)' % (se.upper(),md.upper(),la,lo,yr))
        ax.set_xlabel(r'$T_{500}$ (K)')
        ax.set_ylabel(r'$T_{\mathrm{2\,m}}$ (K)')
        # plt.legend()
        plt.tight_layout()
        plt.savefig('%s/scatter.kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
        plt.close()

