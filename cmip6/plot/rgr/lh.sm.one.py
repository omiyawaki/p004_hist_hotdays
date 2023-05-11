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

# regression model
rmethod='wgtlogi'

# index for location to make plot
iloc=[110,85] # SEA
# iloc=[135,200] # SWUS

varn1='hfls'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='historical'
yr0='2000-2022' # hydroclimate regime years
yr='1980-2000'
lmd=mods(fo)

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

# regression function
def modlogifunc(x,A,x0,k,x2,k2):
    return A / (1 + np.exp(-k*(x-x0))) / (1 + np.exp(-k2*(x-x2)))
# def logifunc(x,A,x0,k,off):
#         return A / (1 + np.exp(-k*(x-x0)))+off
def logifunc(x,A,x0,k):
        return A / (1 + np.exp(-k*(x-x0)))

for se in lse:
    idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
    # load hydroclimate regime info
    [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
    hr=hr*lm
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    for md in tqdm(lmd):
        print(md)

        # scatter data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)

        fn1 = '%s/cl%s_%s.%s.nc' % (idir1,varn1,yr,se)
        fn2 = '%s/cl%s_%s.%s.nc' % (idir2,varn2,yr,se)

        ds1=xr.open_dataset(fn1)
        vn1=ds1[varn1].load()
        ds2=xr.open_dataset(fn2)
        vn2=ds2[varn2].load()
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

        lsm=np.linspace(np.nanmin(l2),np.nanmax(l2),500)
        msm,mlh=np.mgrid[np.nanmin(l2):np.nanmax(l2):100j,np.nanmin(l1):np.nanmax(l1):250j]
        abm=np.vstack([msm.ravel(),mlh.ravel()])

        # kde data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

        fn = '%s/k%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        kde = pickle.load(open(fn,'rb'))
        pdf=np.reshape(kde(abm).T,msm.shape)

        # load regression data
        fn = '%s/r%s%s_%s.%g.%g.%s.pickle' % (idir,rmethod,varn,yr,iloc[0],iloc[1],se)
        popt = pickle.load(open(fn,'rb'))
        if rmethod=='mlogi':
            llh=modlogifunc(lsm,popt[0],popt[1],popt[2],popt[3],popt[4])
        else:
            # llh=logifunc(lsm,popt[0],popt[1],popt[2],popt[3])
            llh=logifunc(lsm,popt[0],popt[1],popt[2])

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # plot scatter of var 1 and var2
        fig,ax=plt.subplots(figsize=(5,4))
        ax.axhline(0,color='k',linewidth=0.5)
        ax.scatter(l2,l1,c=cm[lhr],s=0.5,label=hrn[lhr])
        xlim=ax.get_xlim()
        ylim=ax.get_ylim()
        ax.contour(msm,mlh,pdf)
        ax.plot(lsm,llh,'-k')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(r'%s %s [%+05.1f,%+05.1f] (%s)' % (se.upper(),md.upper(),la,lo,yr))
        ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax.set_ylabel(r'$LH$ (W m$^{-2}$)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/scatter.rgr.%s.kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,rmethod,varn,la,lo,se), format='pdf', dpi=300)
        plt.close()

