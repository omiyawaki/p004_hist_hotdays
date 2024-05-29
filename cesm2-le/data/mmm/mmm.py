import os
import sys
sys.path.append('../../data/')
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
from scipy.stats import linregress
from tqdm import tqdm
# from util import mods

lvn=['tas'] # input1
se='ts'
tavg='jja'
fo='lens2'
yr='1950-2020'

lmd=['CESM2.%03d'%i for i in np.arange(1,101)]

def calc_mmm(varn):

    for i,md in enumerate(tqdm(lmd)):
        idir0 = '/project/amp02/miyawaki/data/p004/cesm2-le/%s/%s/%s/%s' % (se,fo,md,varn)
        odir0 = '/project/amp02/miyawaki/data/p004/cesm2-le/%s/%s/%s/%s' % (se,fo,md,varn)
        if not os.path.exists(odir0):
            os.makedirs(odir0)

        # percentile trend
        pvn=pickle.load(open('%s/trend.pc.%s.%s.%s.pickle' % (idir0,varn,yr,tavg),'rb'))
        pvn=np.stack([pvn[i]['slope'] for i in range(len(pvn))])

        # mean trend
        mvn=pickle.load(open('%s/trend.m.%s.%s.%s.pickle' % (idir0,varn,yr,tavg),'rb'))['slope']

        # relative trend
        ddpvn=pvn-np.transpose(mvn[...,None],[1,0])

        if i==0:
            imvn=np.empty(np.insert(np.asarray(mvn.shape),0,len(lmd)))
            ipvn=np.empty(np.insert(np.asarray(pvn.shape),0,len(lmd)))
            iddpvn=np.empty(np.insert(np.asarray(ddpvn.shape),0,len(lmd)))

        imvn[i,...]=mvn
        ipvn[i,...]=pvn
        iddpvn[i,...]=ddpvn

        # save individual model data
        ddpvn=[ddpvn[i,:] for i in range(ddpvn.shape[0])]
        pickle.dump(ddpvn,open('%s/trend.ddpc.%s.%s.%s.pickle' % (odir0,varn,yr,tavg),'wb'))

    # compute mmm and std
    mmvn=mvn.copy()
    mmvn=np.nanmean(imvn,axis=0)

    mpvn=pvn.copy()
    mpvn=np.nanmean(ipvn,axis=0)

    mddpvn=mpvn.copy()
    mddpvn=np.nanmean(iddpvn,axis=0)

    smvn=mmvn.copy()
    smvn=np.nanstd(imvn,axis=0)

    spvn=mpvn.copy()
    spvn=np.nanstd(ipvn,axis=0)

    sddpvn=mpvn.copy()
    sddpvn=np.nanstd(iddpvn,axis=0)

    # save mmm and std
    odir = '/project/amp02/miyawaki/data/p004/cesm2-le/%s/%s/%s/%s' % (se,fo,'mmm',varn)
    if not os.path.exists(odir):
        os.makedirs(odir)

    mpvn=[mpvn[i,:] for i in range(mpvn.shape[0])]
    spvn=[spvn[i,:] for i in range(spvn.shape[0])]

    mddpvn=[mddpvn[i,:] for i in range(mddpvn.shape[0])]
    sddpvn=[sddpvn[i,:] for i in range(sddpvn.shape[0])]

    pickle.dump(mmvn,open('%s/trend.m.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(smvn,open('%s/std.trend.m.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

    pickle.dump(mpvn,open('%s/trend.pc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(spvn,open('%s/std.trend.pc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

    pickle.dump(mddpvn,open('%s/trend.ddpc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))
    pickle.dump(sddpvn,open('%s/std.trend.ddpc.%s.%s.%s.pickle' % (odir,varn,yr,tavg),'wb'))

[calc_mmm(vn) for vn in lvn]
