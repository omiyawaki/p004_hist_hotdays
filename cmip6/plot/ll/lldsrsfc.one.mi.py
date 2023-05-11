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

varn1='rsfc'
varn2='thd'
varn='%s+%s'%(varn1,varn2)

ivarn01='rsds'
ivarn1='%s+%s'%(ivarn01,varn2)
ivarn02='rsus'
ivarn2='%s+%s'%(ivarn02,varn2)
ivarn03='rlds'
ivarn3='%s+%s'%(ivarn03,varn2)
ivarn04='rlus'
ivarn4='%s+%s'%(ivarn04,varn2)

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
vnt=90
lpc=[95]

lmd=mods(fo1)

for pc in lpc:
    for se in lse:
        c=0
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)

            # load varn1 historical climatology
            idir0 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,'pr')
            [c1, gr] = pickle.load(open('%s/c%s_%s.%s.rg.pickle' % (idir0,'pr',yr1,se), 'rb'))
            ila=iloc[0]
            ilo=iloc[1]
            la=gr['lat'][ila]
            lo=gr['lon'][ilo]
            # mean
            m1=c1[0,ila,ilo]*86400

            # load llds composite
            idir1p1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,ivarn1)
            idir2p1='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,ivarn1)
            [llds1p1,std1p1]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir1p1,ivarn1,nt,yr1,pc,se),'rb'))
            [llds2p1,std2p1]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir2p1,ivarn1,nt,yr2,pc,se),'rb'))
            llds1p1=llds1p1[:,ila,ilo]
            llds2p1=llds2p1[:,ila,ilo]
            std1p1=std1p1[:,ila,ilo]
            std2p1=std2p1[:,ila,ilo]

            idir1p2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,ivarn2)
            idir2p2='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,ivarn2)
            [llds1p2,std1p2]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir1p2,ivarn2,nt,yr1,pc,se),'rb'))
            [llds2p2,std2p2]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir2p2,ivarn2,nt,yr2,pc,se),'rb'))
            llds1p2=llds1p2[:,ila,ilo]
            llds2p2=llds2p2[:,ila,ilo]
            std1p2=std1p2[:,ila,ilo]
            std2p2=std2p2[:,ila,ilo]

            idir1p3='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,ivarn3)
            idir2p3='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,ivarn3)
            [llds1p3,std1p3]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir1p3,ivarn3,nt,yr1,pc,se),'rb'))
            [llds2p3,std2p3]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir2p3,ivarn3,nt,yr2,pc,se),'rb'))
            llds1p3=llds1p3[:,ila,ilo]
            llds2p3=llds2p3[:,ila,ilo]
            std1p3=std1p3[:,ila,ilo]
            std2p3=std2p3[:,ila,ilo]

            idir1p4='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl1,fo1,md,ivarn4)
            idir2p4='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl2,fo2,md,ivarn4)
            [llds1p4,std1p4]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir1p4,ivarn4,nt,yr1,pc,se),'rb'))
            [llds2p4,std2p4]=pickle.load(open('%s/llds%s%03d_%s.%g.%s.pickle' % (idir2p4,ivarn4,nt,yr2,pc,se),'rb'))
            llds1p4=llds1p4[:,ila,ilo]
            llds2p4=llds2p4[:,ila,ilo]
            std1p4=std1p4[:,ila,ilo]
            std2p4=std2p4[:,ila,ilo]

            # compute swsfc
            llds1=llds1p1-llds1p2+llds1p3-llds1p4
            llds2=llds2p1-llds2p2+llds2p3-llds2p4
            std1=np.sqrt(1/4*(std1p1**2+std1p2**2+std1p3**2+std1p4**2))
            std2=np.sqrt(1/4*(std2p1**2+std2p2**2+std2p3**2+std2p4**2))

            dc=np.floor(llds1.shape[0]/2)
            dt=np.arange(llds1.shape[0])-dc
            sl=np.arange(-vnt,vnt+1)+int(dc)

            if c==0:
                odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)

                if not os.path.exists(odir):
                    os.makedirs(odir)

                fig1,ax1=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
                ax1=ax1.flatten()
                fig1.suptitle(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
                fig1.supxlabel('Lag from $T^{>%g}$ event (days)'%(pc))
                fig1.supylabel(r'Deseasonalized $R^{net}_{SFC}$ (W m$^{-2}$)')

                fig2,ax2=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
                ax2=ax2.flatten()
                fig2.suptitle(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
                fig2.supxlabel('Lag from $T^{>%g}$ event (days)'%(pc))
                fig2.supylabel(r'Deseasonalized $R^{net}_{SFC}$ (W m$^{-2}$)')

                fig3,ax3=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
                ax3=ax3.flatten()
                fig3.suptitle(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
                fig3.supxlabel('Lag from $T^{>%g}$ event (days)'%(pc))
                fig3.supylabel(r'$\Delta R^{net}_{SFC}$ (W m$^{-2}$)')

                c=1

            # plot historical llds
            ax1[imd].axhline(0,color='k',linewidth=0.5)
            ax1[imd].axvline(0,color='k',linewidth=0.5)
            ax1[imd].fill_between(dt[sl],llds1[sl]-std1[sl],llds1[sl]+std1[sl],color=co1,edgecolor=None,alpha=0.2)
            ax1[imd].plot(dt[sl],llds1[sl],color=co1)
            ax1[imd].set_xlim([-vnt,vnt])
            ax1[imd].set_title(r'%s' % (md))
            fig1.tight_layout()
            fig1.savefig('%s/llds.hist.%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)

            # plot historical llds
            ax2[imd].axhline(0,color='k',linewidth=0.5)
            ax2[imd].axvline(0,color='k',linewidth=0.5)
            ax2[imd].fill_between(dt[sl],llds2[sl]-std2[sl],llds2[sl]+std2[sl],color=co2,edgecolor=None,alpha=0.2)
            ax2[imd].plot(dt[sl],llds2[sl],color=co2)
            ax2[imd].set_xlim([-vnt,vnt])
            ax2[imd].set_title(r'%s' % (md))
            fig2.tight_layout()
            fig2.savefig('%s/llds.fut.%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)

            # plot future - historical llds
            dllds=llds2[sl]-llds1[sl]
            dstd=np.sqrt(std1[sl]**2+std2[sl]**2)
            ax3[imd].axhline(0,color='k',linewidth=0.5)
            ax3[imd].axvline(0,color='k',linewidth=0.5)
            ax3[imd].fill_between(dt[sl],dllds-dstd,dllds+dstd,color='k',edgecolor=None,alpha=0.2)
            ax3[imd].plot(dt[sl],dllds,color='k')
            ax3[imd].set_xlim([-vnt,vnt])
            ax3[imd].set_title(r'%s' % (md))
            fig3.tight_layout()
            fig3.savefig('%s/llds.fut-hist.d%s%03d.%g.%s.pdf' % (odir,varn,vnt,pc,se), format='pdf', dpi=300)

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
