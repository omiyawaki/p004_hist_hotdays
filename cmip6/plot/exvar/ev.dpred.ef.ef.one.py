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

md='mi'
varn1='ef'
varn2='ef'
varn='%s+%s'%(varn1,varn2)
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
his='1980-2000'
fut='2080-2100'
lpc=[0,95,99]

for pc in lpc:
    for se in lse:

        idir='/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical','CESM2','hd')
        [_,gr]=pickle.load(open('%s/cd%s_%s.%g.%s.pickle' % (idir,'hd',his,95,se),'rb'))
        la=gr['lat'][iloc[0]]
        lo=gr['lon'][iloc[1]]

        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)

        odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,md,varn,la,lo)

        if not os.path.exists(odir):
            os.makedirs(odir)

        ev=pickle.load(open('%s/ev.dpred.%s.%s.%s.%g.%s.pickle' % (idir,varn,his,fut,pc,se),'rb'))

        # exvar bar plot 
        fig,ax=plt.subplots(figsize=(4,3))

        xl=range(len(ev)-1)
        ax.bar(xl,ev[1:],color='k')
        ax.set_ylim([0,1])
        ax.set_xticks(xl)
        ax.set_xticklabels([r'$\Delta BC$',r'$BC_H$',r'$\Delta SM$',r'$SM_H$'])
        if pc==0:
            ax.set_ylabel(r'Explained Intermodel Variance of $\Delta \overline{EF}$')
        else:
            ax.set_ylabel(r'Explained Intermodel Variance of $\Delta EF^{%g}$'%pc)
        ax.set_title(r'%s [%+05.1f,%+05.1f]' % (se.upper(),la,lo))
        plt.tight_layout()
        plt.savefig('%s/ev.dpred.%s.%g.%s.pdf' % (odir,varn,pc,se), format='pdf', dpi=300)
        plt.close()

