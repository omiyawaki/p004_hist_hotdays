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

varn1='mrsos'
varn2='thd'
varn='%s+%s'%(varn1,varn2)
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo1='historical'
yr1='1980-2000'
co1='tab:blue'

fo2='ssp370'
yr2='2080-2100'
co2='tab:orange'

fo='%s+%s'%(fo1,fo2)
cl='%s+%s'%(cl1,cl2)
yr='%s+%s'%(yr1,yr2)

nt=90
lvnt=[7,14,30,90]
lpc=[95]

lmd=mods(fo1)

for pc in lpc:
    for se in lse:
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)

            # load varn1 historical climatology
            idir0 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn1)
            [c1, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir0,varn1,yr1,se), 'rb'))
            ila=iloc[0]
            ilo=iloc[1]
            la=gr['lat'][ila]
            lo=gr['lon'][ilo]
            # mean
            m1=c1[0,ila,ilo]

            # load ll composite
            idir1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,varn)
            idir2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,varn)
            [ll1,std1]=pickle.load(open('%s/ll%s%03d_%s.%g.%s.pickle' % (idir1,varn,nt,yr1,pc,se),'rb'))
            [ll2,std2]=pickle.load(open('%s/ll%s%03d_%s.%g.%s.pickle' % (idir2,varn,nt,yr2,pc,se),'rb'))
            ll1=ll1[:,ila,ilo]
            ll2=ll2[:,ila,ilo]
            std1=std1[:,ila,ilo]
            std2=std2[:,ila,ilo]

            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            for vnt in lvnt:
                dc=np.floor(ll1.shape[0]/2)
                dt=np.arange(ll1.shape[0])-dc
                sl=np.arange(-vnt,vnt+1)+int(dc)

                # plot historical ll
                fig,ax=plt.subplots(figsize=(5,4))
                ax.axhline(m1,color=co1,linewidth=0.5)
                ax.fill_between(dt[sl],ll1[sl]-std1[sl],ll1[sl]+std1[sl],color=co1,edgecolor=None,alpha=0.2)
                ax.plot(dt[sl],ll1[sl],color=co1)
                ax.set_xlim([-vnt,vnt])
                ax.set_xlabel('Lead-lag (days)')
                ax.set_ylabel(r'$SM$ (kg m$^{-2}$)')
                ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (md,se.upper(),la,lo))
                plt.tight_layout()
                plt.savefig('%s/ll.hist.%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)
                plt.close()

                # plot historical del ll
                fig,ax=plt.subplots(figsize=(5,4))
                ax.axhline(0,color=co1,linewidth=0.5)
                ax.fill_between(dt[sl],ll1[sl]-std1[sl]-m1,ll1[sl]+std1[sl]-m1,color=co1,edgecolor=None,alpha=0.2)
                ax.plot(dt[sl],ll1[sl]-m1,color=co1)
                ax.set_xlim([-vnt,vnt])
                ax.set_xlabel('Lead-lag (days)')
                ax.set_ylabel(r'$\delta SM$ (kg m$^{-2}$)')
                ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (md,se.upper(),la,lo))
                plt.tight_layout()
                plt.savefig('%s/ll.hist.d%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)
                plt.close()

                # plot future del ll
                fig,ax=plt.subplots(figsize=(5,4))
                ax.axhline(0,color=co2,linewidth=0.5)
                ax.fill_between(dt[sl],ll2[sl]-std2[sl]-m1,ll2[sl]+std2[sl]-m1,color=co2,edgecolor=None,alpha=0.2)
                ax.plot(dt[sl],ll2[sl]-m1,color=co2)
                ax.set_xlim([-vnt,vnt])
                ax.set_xlabel('Lead-lag (days)')
                ax.set_ylabel(r'$\delta SM$ (kg m$^{-2}$)')
                ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (md,se.upper(),la,lo))
                plt.tight_layout()
                plt.savefig('%s/ll.fut.d%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)
                plt.close()

                # plot future - historical del ll
                dll=ll2[sl]-ll1[sl]
                dstd=np.sqrt(std1[sl]**2+std2[sl]**2)
                fig,ax=plt.subplots(figsize=(5,4))
                ax.axhline(0,color='k',linewidth=0.5)
                ax.fill_between(dt[sl],dll-dstd,dll+dstd,color='k',edgecolor=None,alpha=0.2)
                ax.plot(dt[sl],dll,color='k')
                ax.set_xlim([-vnt,vnt])
                ax.set_xlabel('Lead-lag (days)')
                ax.set_ylabel(r'$\Delta SM$ (kg m$^{-2}$)')
                ax.set_title(r'%s %s [%+05.1f,%+05.1f]' % (md,se.upper(),la,lo))
                plt.tight_layout()
                plt.savefig('%s/ll.fut-hist.d%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)
                plt.close()

