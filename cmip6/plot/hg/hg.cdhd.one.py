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

# fo='historical'
# yr='1980-2000'
# co='tab:blue'

fo='ssp370'
yr='2080-2100'
co='tab:orange'

lpc=[95,99]

lmd=mods(fo)

for pc in lpc:
    for se in lse:
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]
            print(md)

            idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
            [dhd,gr]=pickle.load(open('%s/cd%s_%s.%g.%s.pickle' % (idir,varn,yr,pc,se),'rb'))
            la=gr['lat'][iloc[0]]
            lo=gr['lon'][iloc[1]]

            odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

            if not os.path.exists(odir):
                os.makedirs(odir)

            # plot histogram
            fig,ax=plt.subplots(figsize=(5,4))

            be=np.arange(dhd.max())+1
            ax.hist(dhd,bins=be,color=co)
            ax.set_xlim([be[0],be[-1]])
            ax.set_xticks(be+0.5)
            ax.set_xticklabels(be.astype(int))
            ax.set_xlabel('Days')
            ax.set_ylabel('Frequency')
            ax.set_title(r'Consecutive days exceeding $T^{%g}$' '\n' r'%s %s [%+05.1f,%+05.1f] (%s)' % (pc,md,se.upper(),la,lo,yr))
            plt.tight_layout()
            plt.savefig('%s/hg.cd%s.%g.%s.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
            plt.close()

