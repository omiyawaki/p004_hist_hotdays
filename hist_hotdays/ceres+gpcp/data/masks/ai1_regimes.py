import os,sys
import pickle
import numpy as np
import xarray as xr

varn='hr'
# lse = ['jja'] # season (ann, djf, mam, jja, son)
lse = ['ann','jja','djf','mam','son'] # season (ann, djf, mam, jja, son)
y0=2000 # begin analysis year
y1=2022 # end analysis year
pr0=5.5 # cold vs tropical humid threshold in mm/d

tyr=np.arange(y0,y1)
lyr=[str(y) for y in tyr]

for se in lse:
    odir='/project/amp/miyawaki/data/p004/hist_hotdays/ceres+gpcp/%s/%s' % (se,varn)

    if not os.path.exists(odir):
        os.makedirs(odir)

    # load aridity index data
    ai,gr=pickle.load(open('%s/../ai1/clmean.mai1.%g-%g.%s.pickle' % (odir,y0,y1,se), 'rb'))

    # load precip data
    pr,_=pickle.load(open('%s/../pr/clmean.pr.%g-%g.%s.pickle' % (odir,y0,y1,se), 'rb'))
    pr=pr*86400 # convert to mm/d

    # define hydroclimate regions following Takeshima et al (2020)
    hr=np.empty_like(ai)
    # cold humid
    ch=np.where(np.logical_and(ai<=0.7,pr<=pr0))
    hr[ch]=0
    # tropical humid
    th=np.where(np.logical_and(ai<=0.7,pr>pr0))
    hr[th]=1
    # humid
    h=np.where(np.logical_and(ai<=1.2,ai>0.7))
    hr[h]=2
    # semihumid
    sh=np.where(np.logical_and(ai<=2.0,ai>1.2))
    hr[sh]=3
    # semiarid
    sa=np.where(np.logical_and(ai<=4.0,ai>2.0))
    hr[sa]=4
    # arid
    a=np.where(np.logical_and(ai<=6.0,ai>4.0))
    hr[a]=5
    # hyperarid
    ha=np.where(ai>6.0)
    hr[ha]=6

    pickle.dump([hr, gr], open('%s/%s.%g-%g.%s.pickle' % (odir,varn,y0,y1,se), 'wb'), protocol=5)	
