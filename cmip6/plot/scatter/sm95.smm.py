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

varn1='mrsos'
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
        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)

        if not os.path.exists(odir):
            os.makedirs(odir)

        # load hydroclimate regime info
        [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
        hr=hr*lm
        if reg!='all':
            hr=sellatlon(hr,gr,reg)
        hr=hr[:]

        for ipc in range(len(lpc)):
            pc = lpc[ipc]

            if pc==95:
                xpc=1
            elif pc==99:
                xpc=2

            [cq, gr] = pickle.load(open('%s/c%s_%s.%s.pickle' % (idir1,varn1,yr,se), 'rb'))
            cq=cq['mean']
            cx=cq[0]
            cy=cq[xpc]

            cx=cx*lm
            cy=cy*lm
            if reg!='all':
                cx=sellatlon(cx,gr,reg)
                cy=sellatlon(cy,gr,reg)
            # flatten
            cx=cx[:]
            cy=cy[:]

            fig,ax=plt.subplots(figsize=(5,4))
            for ihr in range(7):
                selhr=np.where(hr==ihr)
                shr=hr[selhr]
                scx=cx[selhr]
                scy=cy[selhr]
                # means
                mscx=np.nanmean(scx)
                mscy=np.nanmean(scy)

                # plot q2m deficit on hot days against t2m
                # ax.axhline(0,color='k',linewidth=0.5)
                ax.scatter(scx,scy,c=cm[ihr],s=0.5,zorder=1)
                ax.plot(mscx,mscy,'s',color=cm[ihr],markersize=5,markeredgecolor='k',zorder=3,label=hrn[ihr])

            lims=[[0,0],[100,100]]
            # lims = [
            #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            #     ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_title(r'%s %s %s (%s)' % (se.upper(),md.upper(),reg.upper(),yr))
            ax.set_xlabel(r'$\overline{SM}_\mathrm{2\,m}$ (kg m$^{-2}$)')
            ax.set_ylabel(r'$SM^{>%g}_\mathrm{2\,m}$ (kg m$^{-2}$)' % (pc))
            plt.legend(prop={'size':8})
            plt.tight_layout()
            plt.savefig('%s/scatter.%s.mean.%02d.%s.%s.pdf' % (odir,varn1,pc,se,reg), format='pdf', dpi=300)
            plt.close()

