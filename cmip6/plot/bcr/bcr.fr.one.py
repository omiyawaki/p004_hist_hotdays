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
    for imd in tqdm(range(len(lmd))):
        md=lmd[imd]
        print(md)

        # bcr data
        idir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn)
        fn = '%s/bcr%s_%s.%g.%g.%s.pickle' % (idir,varn,yr,iloc[0],iloc[1],se)
        [th,fr] = pickle.load(open(fn,'rb'))
        frmi[:,imd]=fr

    # reorder data in water limited regime descending order
    iro=np.argsort(-frmi[0,:])
    # sorted data
    slmd=np.take(lmd,iro)
    sfrmi=np.take(frmi,iro,axis=1)
    # sort data in water saturated regime ascending order where dry is absent
    nd=np.where(sfrmi[0,:]==0)[0]
    ndlmd=np.take(slmd,nd)
    ndfrmi=np.take(sfrmi,nd,axis=1)
    iro2=np.argsort(ndfrmi[2,:])
    # sorted data again
    s2lmd=np.take(ndlmd,iro2)
    s2frmi=np.take(ndfrmi,iro2,axis=1)
    slmd[nd]=s2lmd
    sfrmi[:,nd]=s2frmi

    # plot stacked bar
    fig,ax=plt.subplots(figsize=(8,4))

    ax.bar(slmd,sfrmi[0,:],width=0.5,label='Dry',color='tab:orange')
    ax.bar(slmd,sfrmi[1,:],width=0.5,label='Mix',color='gray',bottom=sfrmi[0,:])
    ax.bar(slmd,sfrmi[2,:],width=0.5,label='Wet',color='tab:blue',bottom=sfrmi[0,:]+sfrmi[1,:])
    ax.set_title(r'%s [%+05.1f,%+05.1f] (%s)' % (se.upper(),la,lo,yr))
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/bcr.fr.%s.kde.%s.%+05.1f.%+05.1f.%s.pdf' % (odir,rmethod,varn,la,lo,se), format='pdf', dpi=300)
    plt.close()

