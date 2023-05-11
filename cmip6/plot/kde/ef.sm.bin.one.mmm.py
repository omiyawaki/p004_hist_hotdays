import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from cmip6util import mods

# index for location to make plot
iloc=[110,85] # SEA
# iloc=[135,200] # SWUS

psm=np.arange(100)+1

varn1='ef'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='historical'
yr='1980-2000'

# fo='ssp370'
# yr='2080-2100'

yr0='2000-2022' # hydroclimate regime years
lmd=mods(fo)
cs=cm.rainbow(np.linspace(0, 1, len(lmd)))

cm=['indigo','darkgreen','royalblue','limegreen','yellow','orangered','maroon']
hrn=['Cold Humid','Tropical Humid','Humid','Semihumid','Semiarid','Arid','Hyperarid']

# load land mask
lm,_=pickle.load(open('/project/amp/miyawaki/data/share/lomask/cesm2/lomask.pickle','rb'))

for se in lse:
    idir0='/project/amp/miyawaki/data/p004/ceres+gpcp/%s/%s'%(se,'hr')
    # load hydroclimate regime info
    [hr, gr] = pickle.load(open('%s/%s.%s.%s.pickle' % (idir0,'hr',yr0,se), 'rb'))
    hr=hr*lm
    la=gr['lat'][iloc[0]]
    lo=gr['lon'][iloc[1]]

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mmm',varn,la,lo)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # plot scatter of var 1 and var2
    fig,ax=plt.subplots(figsize=(5,3))

    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        # binned data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        bef,gr = pickle.load(open('%s/b%s_%s.%s.pickle' % (idir,varn,yr,se), 'rb'))
        ila=iloc[0]
        ilo=iloc[1]
        la=gr['lat'][ila]
        lo=gr['lon'][ilo]
        bef=bef[:,ila,ilo]
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,'mrsos+mrsos')
        bsm,_ = pickle.load(open('%s/b%s_%s.%s.pickle' % (idir,'mrsos+mrsos',yr,se), 'rb'))
        bsm=bsm[:,ila,ilo]

        ax.axhline(0,color='k',linewidth=0.5)
        ax.plot(bsm,bef,'.',color=cs[imd],markersize=2,label=md)
        ax.set_xlim([0,50])
        ax.set_ylim([0,1.5])
        ax.set_title(r'%s [%+05.1f,%+05.1f] (%s)' % (se.upper(),la,lo,yr))
        ax.set_xlabel(r'$SM$ (kg m$^{-2}$)')
        ax.set_ylabel(r'$EF$ (unitless)')
        plt.tight_layout()
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc='center left',bbox_to_anchor=(1, 0.5),prop={'size':7},frameon=False)
        plt.savefig('%s/scatter.%s.bin.%+05.1f.%+05.1f.%s.pdf' % (odir,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

