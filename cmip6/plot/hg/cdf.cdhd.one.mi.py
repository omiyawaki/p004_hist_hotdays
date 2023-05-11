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

varn='hd'
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='historical'
yr='1980-2000'
co='tab:blue'

# fo='ssp370'
# yr='2080-2100'
# co='tab:orange'

lpc=[95,99]

lmd=mods(fo)

for pc in lpc:
    for se in lse:

        c=0
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)

            idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            [dhd,gr]=pickle.load(open('%s/cd%s_%s.%g.%s.pickle' % (idir,varn,yr,pc,se),'rb'))
            la=gr['lat'][iloc[0]]
            lo=gr['lon'][iloc[1]]

            if c==0:
                # plot histogram
                fig,ax=plt.subplots(nrows=4,ncols=5,figsize=(12,8))
                ax=ax.flatten()
                fig.suptitle(r'Duration of $>T^{%g}$ events' '\n' '%s [%+05.1f,%+05.1f] (%s)' % (pc,se.upper(),la,lo,yr))

                odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)
                if not os.path.exists(odir):
                    os.makedirs(odir)
                c=1

            be=np.arange(dhd.max())+1
            ax[imd].axhline(0.9,color='k',linewidth=1)
            ax[imd].hist(dhd,bins=be,color=co,cumulative=1,density=True,histtype='step')
            ax[imd].set_xlim([be[0],20])
            ax[imd].set_xticks(be+0.5)
            ax[imd].set_xticklabels(be.astype(int),fontsize=5)
            ax[imd].set_xlabel('Days')
            ax[imd].set_ylabel('Probability')
            ax[imd].set_title(r'%s'%md)
            plt.tight_layout()
            plt.savefig('%s/cdf.cd%s.%g.%s.pdf' % (odir,varn,pc,se), format='pdf', dpi=128)
        plt.close()
