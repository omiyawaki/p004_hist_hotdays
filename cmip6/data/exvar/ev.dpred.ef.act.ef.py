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

varn1='ef' # var
varn2='ef' # var
varn='%s+%s'%(varn1,varn2)
lse = ['jja'] # season (ann, djf, mam, jja, son)

fo='ssp370'
cl='fut-his'
his='1980-2000'
fut='2080-2100'
lpc=[0,95,99]
lmd=mods(fo)

for pc in lpc:
    for se in lse:

        odir = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,'mi',varn1)
        if not os.path.exists(odir):
            os.makedirs(odir)

        for imd in tqdm(range(len(lmd))):
            md=lmd[imd]

            idir1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,cl,fo,md,varn1)
            idir2c1 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'his','historical',md,varn2)
            idir2c2 = '/project/amp/miyawaki/data/p004/cmip6/%s/%s/%s/%s/%s' % (se,'fut','ssp370',md,varn2)

            [vn0, gr] = pickle.load(open('%s/diff_%02d.%s.%s.%s.pickle' % (idir1,pc,his,fut,se), 'rb'))
            vn1c1 = pickle.load(open('%s/pc%s_%s.%s.pickle' % (idir2c1,varn2,his,se), 'rb'))
            vn1c2 = pickle.load(open('%s/pc%s_%s.%s.pickle' % (idir2c2,varn2,fut,se), 'rb'))
            # del BC contrib
            vn2c1 = pickle.load(open('%s/pc.devdbc%s_%s.%s.pickle' % (idir2c1,varn2,his,se), 'rb'))
            vn2c2 = pickle.load(open('%s/pc.devdbc%s_%s.%s.pickle' % (idir2c2,varn2,fut,se), 'rb'))
            # hist BC contrib
            vn3c1 = pickle.load(open('%s/pc.devbc%s_%s.%s.pickle' % (idir2c1,varn2,his,se), 'rb'))
            vn3c2 = pickle.load(open('%s/pc.devbc%s_%s.%s.pickle' % (idir2c2,varn2,fut,se), 'rb'))
            # del SM contrib
            vn4c1 = pickle.load(open('%s/pc.devcsm%s_%s.%s.pickle' % (idir2c1,varn2,his,se), 'rb'))
            vn4c2 = pickle.load(open('%s/pc.devcsm%s_%s.%s.pickle' % (idir2c2,varn2,fut,se), 'rb'))
            # hist SM contrib
            vn5c1 = pickle.load(open('%s/pc.devhistcsm%s_%s.%s.pickle' % (idir2c1,varn2,his,se), 'rb'))
            vn5c2 = pickle.load(open('%s/pc.devhistcsm%s_%s.%s.pickle' % (idir2c2,varn2,fut,se), 'rb'))

            l0=vn0
            if pc==0:
                l1=vn1c2[0]-vn1c1[0]
                l21=vn2c2[0]-vn2c1[0]
                l22=vn3c2[0]-vn3c1[0]
                l23=vn4c2[0]-vn4c1[0]
                l24=vn5c2[0]-vn5c1[0]
            elif pc==95:
                l1=vn1c2[1]-vn1c1[1]
                l21=vn2c2[1]-vn2c1[1]
                l22=vn3c2[1]-vn3c1[1]
                l23=vn4c2[1]-vn4c1[1]
                l24=vn5c2[1]-vn5c1[1]
            elif pc==99:
                l1=vn1c2[2]-vn1c1[2]
                l21=vn2c2[2]-vn2c1[2]
                l22=vn3c2[2]-vn3c1[2]
                l23=vn4c2[2]-vn4c1[2]
                l24=vn5c2[2]-vn5c1[2]

            if imd==0:
                xy0=np.empty([2,len(lmd),l0.shape[0],l0.shape[1]])
                xy1=np.empty([2,len(lmd),l0.shape[0],l0.shape[1]])
                xy2=np.empty([2,len(lmd),l0.shape[0],l0.shape[1]])
                xy3=np.empty([2,len(lmd),l0.shape[0],l0.shape[1]])
                xy4=np.empty([2,len(lmd),l0.shape[0],l0.shape[1]])
            xy0[0,imd,:,:]=l0
            xy0[1,imd,:,:]=l1
            xy1[0,imd,:,:]=l0
            xy1[1,imd,:,:]=l21
            xy2[0,imd,:,:]=l0
            xy2[1,imd,:,:]=l22
            xy3[0,imd,:,:]=l0
            xy3[1,imd,:,:]=l23
            xy4[0,imd,:,:]=l0
            xy4[1,imd,:,:]=l24

        rv0=np.empty_like(l0)
        rv1=np.empty_like(l0)
        rv2=np.empty_like(l0)
        rv3=np.empty_like(l0)
        rv4=np.empty_like(l0)
        for ilo in tqdm(range(l0.shape[1])):
            for ila in range(l0.shape[0]):
                try:
                    _,_,rv0[ila,ilo],_,_=linregress(xy0[1,:,ila,ilo],xy0[0,:,ila,ilo])
                except:
                    rv0[ila,ilo]=0
                try:
                    _,_,rv1[ila,ilo],_,_=linregress(xy1[1,:,ila,ilo],xy1[0,:,ila,ilo])
                except:
                    rv1[ila,ilo]=0
                try:
                    _,_,rv2[ila,ilo],_,_=linregress(xy2[1,:,ila,ilo],xy2[0,:,ila,ilo])
                except:
                    rv2[ila,ilo]=0
                try:
                    _,_,rv3[ila,ilo],_,_=linregress(xy3[1,:,ila,ilo],xy3[0,:,ila,ilo])
                except:
                    rv3[ila,ilo]=0
                try:
                    _,_,rv4[ila,ilo],_,_=linregress(xy4[1,:,ila,ilo],xy4[0,:,ila,ilo])
                except:
                    rv4[ila,ilo]=0
        ev=np.empty([5,l0.shape[0],l0.shape[1]])
        ev[0,:,:]=rv0**2
        ev[1,:,:]=rv1**2
        ev[2,:,:]=rv2**2
        ev[3,:,:]=rv3**2
        ev[4,:,:]=rv4**2
        print(ev[:,110,85])

        pickle.dump(ev, open('%s/ev.dpred.%s.act.%s.%s.%g.%s.pickle' % (odir,varn,his,fut,pc,se), 'wb'), protocol=5)	
