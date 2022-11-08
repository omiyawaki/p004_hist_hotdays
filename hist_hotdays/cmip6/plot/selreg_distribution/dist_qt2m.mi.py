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
from regions import rbin,rlev,rtlm
from cmip6util import mods,simu

varn='qt2m'
lpc=['95']
lre=['sea','swus']
lfo=['ssp245']
lcl=['fut']
# lfo=['historical']
# lcl=['his']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)

for pc in lpc:
    for fo in lfo:
        for re in lre:
            for se in lse:
                for cl in lcl:
                    odir='/project2/tas1/miyawaki/projects/000_hotdays/plots/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn)

                    if not os.path.exists(odir):
                        os.makedirs(odir)

                    lmd=mods(fo)

                    sim=simu(fo,cl)
                    if sim=='ssp245':
                        yr='208001-210012'
                    elif sim=='historical':
                        yr='198001-200012'

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
                                [pdf,mt,mq] = pickle.load(open('%s/clmean.k%s.%s.%s.%s.pickle' % (idir,varn,yr,re,se), 'rb'))
                            else:
                                [pdf,mt,mq] = pickle.load(open('%s/clmean.k%s.gt%s.%s.%s.%s.pickle' % (idir,varn,pc,yr,re,se), 'rb'))
                            
                            levs=rlev(re,pc)
                            ax[imd].contour(mt,mq,pdf,levs,vmin=levs[0],vmax=levs[-1])
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
                                plt.savefig('%s/kde.%s.%s.%s.%s.pdf' % (odir,varn,yr,re,se), format='pdf', dpi=300)
                            else:
                                plt.savefig('%s/kde.%s.gt%s.%s.%s.%s.pdf' % (odir,varn,pc,yr,re,se), format='pdf', dpi=300)
                    if pc=='':
                        plt.savefig('%s/kde.%s.%s.%s.%s.pdf' % (odir,varn,yr,re,se), format='pdf', dpi=300)
                    else:
                        plt.savefig('%s/kde.%s.gt%s.%s.%s.%s.pdf' % (odir,varn,pc,yr,re,se), format='pdf', dpi=300)
                    plt.close()
