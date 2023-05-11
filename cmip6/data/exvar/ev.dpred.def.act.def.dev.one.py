import os
import sys
sys.path.append('/home/miyawaki/scripts/common')
sys.path.append('/home/miyawaki/scripts/p004/scripts/cmip6/data')
import pickle
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from cmip6util import mods

# index for location to make plot
iloc=[110,85] # SEA
ila=iloc[0]
ilo=iloc[1]

varn1='ef' # var
varn2='ef' # var
varn='%s+%s'%(varn1,varn2)
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
his='1980-2000'
fut='2080-2100'
lpc=[95,99]
lmd=mods(fo)

for pc in lpc:
    for se in lse:

        odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn1)
        if not os.path.exists(odir):
            os.makedirs(odir)

        xy0=np.empty([2,len(lmd)])
        xy1=np.empty([2,len(lmd)])
        xy2=np.empty([2,len(lmd)])
        xy3=np.empty([2,len(lmd)])
        xy4=np.empty([2,len(lmd)])
        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
            idir2c1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn2)
            idir2c2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'fut','ssp370',md,varn2)

            [vn0p0, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,0,his,fut,se), 'rb'))
            [vn0p1, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,pc,his,fut,se), 'rb'))
            vn1c1 = pickle.load(open('%s/pc%s_%s.%g.%g.%s.dev.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn1c2 = pickle.load(open('%s/pc%s_%s.%g.%g.%s.dev.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))
            # del BC contrib
            vn2c1 = pickle.load(open('%s/pc.devdbc%s_%s.%g.%g.%s.dev.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn2c2 = pickle.load(open('%s/pc.devdbc%s_%s.%g.%g.%s.dev.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))
            # hist BC contrib
            vn3c1 = pickle.load(open('%s/pc.devbc%s_%s.%g.%g.%s.dev.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn3c2 = pickle.load(open('%s/pc.devbc%s_%s.%g.%g.%s.dev.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))
            # del SM contrib
            vn4c1 = pickle.load(open('%s/pc.devcsm%s_%s.%g.%g.%s.dev.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn4c2 = pickle.load(open('%s/pc.devcsm%s_%s.%g.%g.%s.dev.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))
            # hist SM contrib
            vn5c1 = pickle.load(open('%s/pc.devhistcsm%s_%s.%g.%g.%s.dev.pickle' % (idir2c1,varn2,his,ila,ilo,se), 'rb'))
            vn5c2 = pickle.load(open('%s/pc.devhistcsm%s_%s.%g.%g.%s.dev.pickle' % (idir2c2,varn2,fut,ila,ilo,se), 'rb'))

            la=gr['lat'][ila]
            lo=gr['lon'][ilo]

            l0=vn0p1[ila,ilo]-vn0p0[ila,ilo]

            ml1=vn1c2[0]-vn1c1[0]
            ml21=vn2c2[0]-vn2c1[0]
            ml22=vn3c2[0]-vn3c1[0]
            ml23=vn4c2[0]-vn4c1[0]
            ml24=vn5c2[0]-vn5c1[0]
            if pc==95:
                l1=vn1c2[1]-vn1c1[1] -ml1
                l21=vn2c2[1]-vn2c1[1]-ml21
                l22=vn3c2[1]-vn3c1[1]-ml22
                l23=vn4c2[1]-vn4c1[1]-ml23
                l24=vn5c2[1]-vn5c1[1]-ml24
            elif pc==99:
                l1=vn1c2[2]-vn1c1[2] -ml1
                l21=vn2c2[2]-vn2c1[2]-ml21
                l22=vn3c2[2]-vn3c1[2]-ml22
                l23=vn4c2[2]-vn4c1[2]-ml23
                l24=vn5c2[2]-vn5c1[2]-ml24
            xy0[0,imd]=l0
            xy0[1,imd]=l1
            xy1[0,imd]=l0
            xy1[1,imd]=l21
            xy2[0,imd]=l0
            xy2[1,imd]=l22
            xy3[0,imd]=l0
            xy3[1,imd]=l23
            xy4[0,imd]=l0
            xy4[1,imd]=l24

        _,_,rv0,_,_=linregress(xy0[1,:],xy0[0,:])
        _,_,rv1,_,_=linregress(xy1[1,:],xy1[0,:])
        _,_,rv2,_,_=linregress(xy2[1,:],xy2[0,:])
        _,_,rv3,_,_=linregress(xy3[1,:],xy3[0,:])
        _,_,rv4,_,_=linregress(xy4[1,:],xy4[0,:])
        ev=np.empty([5])
        ev[0]=rv0**2
        ev[1]=rv1**2
        ev[2]=rv2**2
        ev[3]=rv3**2
        ev[4]=rv4**2
        print(ev)

        pickle.dump(ev, open('%s/ev.dpred.d%s.act.d%s.%s.%g.%s.dev.pickle' % (odir,varn,his,fut,pc,se), 'wb'), protocol=5)	
