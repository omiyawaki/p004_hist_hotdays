import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from cmip6util import mods

# regression model
rmethod='wgtlogi'

# index for location to make plot
# iloc=[110,85] # SEA
iloc=[135,200] # SWUS

varn1='hfls'
varn2='mrsos'
varn='%s+%s'%(varn1,varn2) # yaxis var
lse = ['jja'] # season (ann, djf, mam, jja, son)

# fo='historical'
# yr='1980-2000'

fo='ssp370'
yr='2080-2100'

yr0='2000-2022' # hydroclimate regime years
lmd=mods(fo)

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

    odir = '/project/amp/miyawaki/plots/p004/cmip6/%s/%s/%s/%s/%s/selpoint/lat_%+05.1f/lon_%+05.1f/' % (se,cl,fo,'mi',varn,la,lo)

    if not os.path.exists(odir):
        os.makedirs(odir)

    frmi=np.empty([3,len(lmd)])
    hamp=np.empty(len(lmd))
    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        # bcr data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        fn = '%s/bcr%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        [th,fr] = pickle.load(open(fn,'rb'))
        frmi[:,imd]=fr

        # amplified warming data
        idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'fut-his','ssp370',md,'tas')
        [ddist, gr] = pickle.load(open('%s/ddist.%s.%02d.%s.%s.%s.pickle' % (idir1,'tas',95,'1980-2000','2080-2100',se), 'rb'))
        fint=interp1d(gr['lat'],ddist)
        ddist=fint(la)
        fint=interp1d(gr['lon'],ddist)
        ddist=fint(lo)
        hamp[imd]=ddist
        # hamp[imd]=ddist[iloc[0],iloc[1]]

    # plot scatter
    fig,ax=plt.subplots(figsize=(8,4))

    ax.scatter(frmi[1,:],hamp,s=0.5)
    ax.set_title(r'%s [%+05.1f,%+05.1f] (%s)' % (se.upper(),la,lo,yr))
    ax.set_xlabel('Fraction of days in Mixed Regime')
    ax.set_ylabel(r'$(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}-(T^{%s}_\mathrm{2\,m}-T^{50}_\mathrm{2\,m})_\mathrm{%s}$ (K)' % (95,'2080-2100',95,'1980-2000'))
    plt.tight_layout()
    plt.savefig('%s/amp.fr.%s.kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,rmethod,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

