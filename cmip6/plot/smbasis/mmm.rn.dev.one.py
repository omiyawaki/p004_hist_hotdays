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

varn='rsfc+dmrsos'
lse = ['jja'] # season (ann, djf, mam, jja, son)
md='mmm'

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


    # kde data
    idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
    idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)

    fn1 = '%s/b%s_%s.%g.%g.%s.dev.pickle' % (idir1,varn,yr1,iloc[0],iloc[1],se)
    fn2 = '%s/b%s_%s.%g.%g.%s.dev.pickle' % (idir2,varn,yr2,iloc[0],iloc[1],se)
    [ef1,sm1,eef1] = pickle.load(open(fn1,'rb'))
    [ef2,sm2,eef2] = pickle.load(open(fn2,'rb'))
    asd1=np.nanstd(eef1,axis=0)
    asd2=np.nanstd(eef2,axis=0)

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # plot pdfs of var (hist and fut)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.axhline(0,color='k',linewidth=0.5)
    # ax.fill_between(sm1,ef1-sd1,ef1+sd1,color='tab:blue',ec=None,alpha=0.3)
    # ax.fill_between(sm2,ef2-sd2,ef2+sd2,color='tab:orange',ec=None,alpha=0.3)
    ax.fill_between(sm1,ef1-asd1,ef1+asd1,color='tab:blue',ec=None,alpha=0.3)
    ax.fill_between(sm2,ef2-asd2,ef2+asd2,color='tab:orange',ec=None,alpha=0.3)
    ax.plot(sm1,ef1,color='tab:blue',label='1980-2000')
    ax.plot(sm2,ef2,color='tab:orange',label='2080-2100')
    ax.set_xlim([-30,20])
    # ax.set_ylim([0,2])
    ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
    ax.set_xlabel(r'$\delta SM$ (kg m$^{-2}$)')
    ax.set_ylabel(r'$R_{SFC}$ (W m$^{-2}$)')
    plt.legend(frameon=False,loc='lower left')
    plt.tight_layout()
    plt.savefig('%s/%s.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

    # historical spaghetti
    fig,ax=plt.subplots(figsize=(5,4))
    ax.axhline(0,color='k',linewidth=0.5)
    for imd in range(eef1.shape[0]):
        ax.plot(sm1,eef1[imd,:],alpha=0.5,linewidth=0.5,color='tab:blue')
    ax.plot(sm1,ef1,color='tab:blue',label='1980-2000')
    ax.set_xlim([-30,20])
    # ax.set_ylim([0,2])
    ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
    ax.set_xlabel(r'$\delta SM$ (kg m$^{-2}$)')
    ax.set_ylabel(r'$R_{SFC}$ (W m$^{-2}$)')
    plt.legend(frameon=False,loc='lower left')
    plt.tight_layout()
    plt.savefig('%s/%s.%+05.1f.%+05.1f.%s.his.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

    # future spaghetti
    fig,ax=plt.subplots(figsize=(5,4))
    ax.axhline(0,color='k',linewidth=0.5)
    for imd in range(eef1.shape[0]):
        ax.plot(sm2,eef2[imd,:],alpha=0.5,linewidth=0.5,color='tab:orange')
    ax.plot(sm2,ef2,color='tab:orange',label='2080-2100')
    ax.set_xlim([-30,20])
    # ax.set_ylim([0,2])
    ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
    ax.set_xlabel(r'$\delta SM$ (kg m$^{-2}$)')
    ax.set_ylabel(r'$R_{SFC}$ (W m$^{-2}$)')
    plt.legend(frameon=False,loc='lower left')
    plt.tight_layout()
    plt.savefig('%s/%s.%+05.1f.%+05.1f.%s.fut.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

    # historical spaghetti
    fig,ax=plt.subplots(figsize=(5,4))
    ax.axhline(0,color='k',linewidth=0.5)
    for imd in range(eef1.shape[0]):
        if lmd[imd]=='CE\delta SM2':
            ax.plot(sm2,eef2[imd,:],color='tab:red',label=lmd[imd])
        elif lmd[imd]=='EC-Earth3':
            ax.plot(sm2,eef2[imd,:],color='tab:green',label=lmd[imd])
        elif lmd[imd]=='KACE-1-0-G':
            ax.plot(sm2,eef2[imd,:],color='tab:purple',label=lmd[imd])
        elif lmd[imd]=='UKE\delta SM1-0-LL':
            ax.plot(sm2,eef2[imd,:],color='tab:brown',label=lmd[imd])
        else:
            ax.plot(sm1,eef1[imd,:],alpha=0.5,linewidth=0.5,color='tab:blue')
    ax.plot(sm1,ef1,color='tab:blue',label='MMM 1980-2000')
    ax.set_xlim([-30,20])
    # ax.set_ylim([0,2])
    ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
    ax.set_xlabel(r'$\delta SM$ (kg m$^{-2}$)')
    ax.set_ylabel(r'$R_{SFC}$ (W m$^{-2}$)')
    plt.legend(frameon=False,loc='lower left')
    plt.tight_layout()
    plt.savefig('%s/sel.%s.%+05.1f.%+05.1f.%s.his.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

    # future spaghetti (pick specific models)
    fig,ax=plt.subplots(figsize=(5,4))
    ax.axhline(0,color='k',linewidth=0.5)
    for imd in range(eef1.shape[0]):
        if lmd[imd]=='CE\delta SM2':
            ax.plot(sm2,eef2[imd,:],color='tab:red',label=lmd[imd])
        elif lmd[imd]=='EC-Earth3':
            ax.plot(sm2,eef2[imd,:],color='tab:green',label=lmd[imd])
        elif lmd[imd]=='KACE-1-0-G':
            ax.plot(sm2,eef2[imd,:],color='tab:purple',label=lmd[imd])
        elif lmd[imd]=='UKE\delta SM1-0-LL':
            ax.plot(sm2,eef2[imd,:],color='tab:brown',label=lmd[imd])
        else:
            ax.plot(sm2,eef2[imd,:],alpha=0.5,linewidth=0.5,color='tab:orange')
    ax.plot(sm2,ef2,color='tab:orange',label='MMM 2080-2100')
    ax.set_xlim([-30,20])
    # ax.set_ylim([0,2])
    ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (se.upper(),md.upper(),la,lo))
    ax.set_xlabel(r'$\delta SM-\overline{\delta SM}$ (kg m$^{-2}$)')
    ax.set_ylabel(r'$R_{SFC}$ (W m$^{-2}$)')
    plt.legend(frameon=False,loc='lower left')
    plt.tight_layout()
    plt.savefig('%s/sel.%s.%+05.1f.%+05.1f.%s.fut.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

