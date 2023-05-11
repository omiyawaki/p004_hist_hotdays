import sys
sys.path.append('/home/miyawaki/scripts/common')
import pickle
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from scipy.stats import linregress
from tqdm import tqdm
from regions import rbin

varn='qt2m'
lpc=[''] # evaluate kde for values exceeding the gt percentile
lre=['swus','sea'] # can be empty
# lse = ['ann','djf','mam','jja','son'] # season (ann, djf, mam, jja, son)
lse = ['jja'] # season (ann, djf, mam, jja, son)
y0 = 2000 # begin analysis year
y1 = 2020 # end analysis year

tyr=np.arange(y0,y1+1)
lyr=[str(y) for y in tyr]

for pc in lpc:
    for re in lre:
        mt,mq=rbin(re)
        # mt,mq=np.mgrid[280:320:250j,0:2.5e-2:250j]
        abm=np.vstack([mt.ravel(),mq.ravel()])
        for se in lse:
            odir = '/project/amp/miyawaki/data/p004/hist_hotdays/era5/%s/%s' % (se,varn)

            # load data
            c0=0 # first loop counter
            for iyr in tqdm(range(len(lyr))):
                yr = lyr[iyr]
                if pc=='':
                    kqt2m = pickle.load(open('%s/k%s_%s.%s.%s.pickle' % (odir,varn,yr,re,se), 'rb'))
                else:
                    kqt2m = pickle.load(open('%s/k%s_%s.gt%s.%s.%s.pickle' % (odir,varn,yr,pc,re,se), 'rb'))
                pdf=np.reshape(kqt2m(abm).T,mt.shape)

                # store data
                if c0 == 0:
                    ykqt2m = np.empty([len(lyr),mt.shape[0],mt.shape[1]])
                    c0 = 1

                ykqt2m[iyr,...] = pdf

            # take mean over time
            mqt2m=np.mean(ykqt2m,axis=0)

            if pc=='':
                pickle.dump([mqt2m,mt,mq], open('%s/clmean.kqt2m.%s.%g.%g.%s.pickle' % (odir,re,y0,y1,se), 'wb'), protocol=5)	
            else:
                pickle.dump([mqt2m,mt,mq], open('%s/clmean.kqt2m.gt%s.%s.%g.%g.%s.pickle' % (odir,pc,re,y0,y1,se), 'wb'), protocol=5)	
