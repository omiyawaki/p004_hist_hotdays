import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress
from tqdm import tqdm
from regions import sellatlon

varn1='huss'
varn2='tas'
lreg=['all','nh','sh'] # region (e.g., all,nh,sh,tr)
lse = ['jja'] # season (ann, djf, mam, jja, son)
# lpc = [1,5,50,95,99] # percentile (choose from lpc below)
lpc = [95] # percentile (choose from lpc below)

md='mmm'
fo='historical'
yr0='2000-2022' # hydroclimate regime years
yr='1980-2000'
cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for reg in lreg:
    for se in lse:
        idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
        idir2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn2)
        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s+%s' % (se,cl,fo,md,varn1,varn2)


        # load hydroclimate regime info
        [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
        hr=hr*lm
        if reg!='all':
            hr=sellatlon(hr,gr,reg)
        hr=hr[:]

        if not os.path.exists(odir):
            os.makedirs(odir)

        for ipc in range(len(lpc)):
            pc = lpc[ipc]
            [cldq, gr] = pickle.load(open('%s/cldist.%s.%02d.%s.%s.pickle' % (idir1,varn1,pc,yr,se), 'rb'))
            [cldt, gr] = pickle.load(open('%s/cldist.%s.%02d.%s.%s.pickle' % (idir2,varn2,pc,yr,se), 'rb'))

            cldq=cldq['mean']
            cldt=cldt['mean']
            cldq=cldq*lm
            cldt=cldt*lm
            if reg!='all':
                cldq=sellatlon(cldq,gr,reg)
                cldt=sellatlon(cldt,gr,reg)
            # flatten
            cldq=cldq[:]
            cldt=cldt[:]

            # plot q2m deficit on hot days against t2m
            fig,ax=plt.subplots(figsize=(5,4))
            ax.axhline(0,color='k',linewidth=0.5)
            for ihr in range(7):
                shr=np.where(hr==ihr)
                scldt=cldt[shr]
                scldq=cldq[shr]
                # means
                mscldt=np.nanmean(scldt)
                mscldq=np.nanmean(scldq)
                ax.scatter(scldt,1e3*scldq,c=cm[ihr],s=0.5,zorder=1)
                ax.plot(mscldt,1e3*mscldq,'s',color=cm[ihr],markersize=5,markeredgecolor='k',zorder=3,label=hrn[ihr])
            ax.set_title(r'%s %s %s (%s)' % (se.upper(),md.upper(),reg.upper(),yr))
            if reg=='nh':
                ax.set_xlim([0,15])
                ax.set_ylim([-10,10])
            else:
                ax.set_xlim([0,25])
                ax.set_ylim([-10,10])
            ax.set_xlabel(r'$T^{>%s}_\mathrm{2\,m}-\overline{T}_\mathrm{2\,m}$ (K)' % (pc))
            ax.set_ylabel(r'$q^{>%s}_\mathrm{2\,m}-\overline{q}_\mathrm{2\,m}$ (g kg$^{-1}$)' % (pc))
            plt.legend(prop={'size':8})
            plt.tight_layout()
            plt.savefig('%s/scatter.%s.%s.%02d.%s.%s.pdf' % (odir,varn1,varn2,pc,se,reg), format='pdf', dpi=300)
            plt.close()

