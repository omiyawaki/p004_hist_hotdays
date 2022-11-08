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
fo1='ssp245'
fo0='historical'
yr1='208001-210012'
yr0='198001-200012'
fo='%s-%s'%(fo1,fo0)
cl='fut-his'
lpc=['95']
lre=['sea']
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lse = ['ann','djf','mam','son','jja'] # season (ann, djf, mam, jja, son)
mmm=True

for pc in lpc:
    for re in lre:
        for se in lse:
                if mmm:
                    lmd=['mmm']
                else:
                    lmd=mods(fo)

                for imd in tqdm(range(len(lmd))):
                    md=lmd[imd]

                    idir='/project2/tas1/miyawaki/projects/000_hotdays/data/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
                    odir='/project2/tas1/miyawaki/projects/000_hotdays/plots/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)

                    if not os.path.exists(odir):
                        os.makedirs(odir)

                    # KDE
                    if mmm:
                        if pc=='':
                            [stats,mt,mq] = pickle.load(open('%s/diff.k%s.%s.%s.%s.%s.pickle' % (idir,varn,yr0,yr1,re,se), 'rb'))
                        else:
                            [stats,mt,mq] = pickle.load(open('%s/diff.k%s.gt%s.%s.%s.%s.%s.pickle' % (idir,varn,pc,yr0,yr1,re,se), 'rb'))
                        dpdf=stats['mean']
                    else:
                        if pc=='':
                            [dpdf,mt,mq] = pickle.load(open('%s/diff.k%s.%s.%s.%s.%s.pickle' % (idir,varn,yr0,yr1,re,se), 'rb'))
                        else:
                            [dpdf,mt,mq] = pickle.load(open('%s/diff.k%s.gt%s.%s.%s.%s.%s.pickle' % (idir,varn,pc,yr0,yr1,re,se), 'rb'))
                    
                    fig,ax=plt.subplots(figsize=(4,3))
                    levs=rlevd(re,pc)
                    ax.contour(mt,mq,dpdf,levs,vmin=levs[0],vmax=levs[-1],cmap='RdBu_r')
                    ax.set_xlim(rtlm(re,pc))
                    ax.set_xlabel(r'$T_{2\,m}$ (K)')
                    ax.set_ylabel(r'$q_{2\,m}$ (kg kg$^{-1}$)')
                    ax.set_title(r'%s %s %s' % (se.upper(),md.upper(),re.upper()))
                    fig.tight_layout()
                    if pc=='':
                        plt.savefig('%s/kde.%s.%s.%s.%s.%s.pdf' % (odir,varn,yr0,yr1,re,se), format='pdf', dpi=300)
                    else:
                        plt.savefig('%s/kde.%s.gt%s.%s.%s.%s.%s.pdf' % (odir,varn,pc,yr0,yr1,re,se), format='pdf', dpi=300)
                    plt.close()
