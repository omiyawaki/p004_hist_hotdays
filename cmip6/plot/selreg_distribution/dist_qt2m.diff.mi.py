import os
import sys
sys.path.append('../../data')
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/project2/tas1/miyawaki/common')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from tqdm import tqdm
from regions import rbin,rlevd,rtlm
from cmip6util import mods,simu

varn='qt2m'
lpc=['95']
lre=['sea','swus']
fo1='ssp245'
fo0='historical'
yr1='208001-210012'
yr0='198001-200012'
fo='%s-%s'%(fo1,fo0)
cl='fut-his'
# lfo=['historical']
# lcl=['his']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)

lmd=mods(fo1)

for pc in lpc:
    for re in lre:
        for se in lse:
            odir='/project2/tas1/miyawaki/projects/000_hotdays/plots/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)

            if not os.path.exists(odir):
                os.makedirs(odir)

            nr=4
            nc=5
            fig,ax=plt.subplots(nrows=nr,ncols=nc,figsize=(11,8.5))
            ax=ax.flatten()
            fig.suptitle(r'%s %s %s'%(se.upper(),fo.upper(),re.upper()))
            # for imd in tqdm(range(len(lmd))):
            for imd in tqdm(range(nr*nc)):
                if imd>len(lmd)-1:
                    ax[imd].set_axis_off()
                else:
                    md=lmd[imd]

                    idir='/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                    # KDE
                    if pc=='':
                        [dpdf,mt,mq] = pickle.load(open('%s/diff.k%s.%s.%s.%s.%s.pickle' % (idir,varn,yr0,yr1,re,se), 'rb'))
                    else:
                        [dpdf,mt,mq] = pickle.load(open('%s/diff.k%s.gt%s.%s.%s.%s.%s.pickle' % (idir,varn,pc,yr0,yr1,re,se), 'rb'))
                    
                    levs=rlevd(re,pc)
                    ax[imd].contour(mt,mq,dpdf,levs,vmin=levs[0],vmax=levs[-1],cmap='RdBu_r')
                    ax[imd].set_xlim(rtlm(re,pc))
                    if np.mod(imd,nc)==0:
                        ax[imd].set_ylabel(r'$q_{2\,m}$ (kg kg$^{-1}$)')
                    else:
                        ax[imd].set_yticks([])
                    if imd>=(nr*(nc-1)-1):
                        ax[imd].set_xlabel(r'$T_{2\,m}$ (K)')
                    else:
                        ax[imd].set_xticks([])
                    ax[imd].set_title(r'%s' % (md.upper()))
                    if pc=='':
                        plt.savefig('%s/kde.%s.%s.%s.%s.%s.pdf' % (odir,varn,yr0,yr1,re,se), format='pdf', dpi=300)
                    else:
                        plt.savefig('%s/kde.%s.gt%s.%s.%s.%s.%s.pdf' % (odir,varn,pc,yr0,yr1,re,se), format='pdf', dpi=300)
            if pc=='':
                plt.savefig('%s/kde.%s.%s.%s.%s.%s.pdf' % (odir,varn,yr0,yr1,re,se), format='pdf', dpi=300)
            else:
                plt.savefig('%s/kde.%s.gt%s.%s.%s.%s.%s.pdf' % (odir,varn,pc,yr0,yr1,re,se), format='pdf', dpi=300)
            plt.close()
