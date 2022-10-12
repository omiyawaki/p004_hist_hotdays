import sys
sys.path.append('/project2/tas1/miyawaki/projects/000_hotdays/scripts')
# from misc.dirnames import get_datadir, get_plotdir
# from misc.filenames import filenames_raw
# from proc.r1 import save_r1
# from plot.titles import make_title_sim_time
import os
import pickle
import glob
import numpy as np
from misc import par
from tqdm import tqdm
from titlestr import make_titlestr
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from netCDF4 import Dataset
# import tikzplotlib

mmm = 1
try_load = 1
proj = '000_hotdays'
# varname = 'mses'
varname1 = 'mses'
varname2 = 'mse50000'
# varname = 'tas'
# modellist = ['UKESM1-0-LL']
# modellist = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
modellist = ['ACCESS-CM2', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'UKESM1-0-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'GFDL-CM4', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
yr_span_hist = '198001-200012'
yr_span_rcp = '208001-210012'
imethod = 'linear'
# regionlist = ['trop', 'nhmid']
regionlist = ['trop']
# seaslist = [None, 'jja', 'son', 'djf', 'mam'] # None for all data points, jja, djf, son, mam
seaslist = [None] # None for all data points, jja, djf, son, mam
prct = np.concatenate((np.array([0.001, 1.]), np.arange(5.,81.,5.), np.arange(82.5,98,2.5), [99]))

# specify latitude bounds for select regions
lat_bounds = {}
lat_bounds['trop'] = {}
lat_bounds['trop']['lo'] = -20
lat_bounds['trop']['up'] = 20
# lat_bounds['nhmid']['lo'] = 20
# lat_bounds['nhmid']['up'] = 40
lat_bounds['nhmid'] = {}
lat_bounds['nhmid']['lo'] = 40
lat_bounds['nhmid']['up'] = 60

# loop over models
for model in modellist:
    if mmm:
        model = 'mmm'

    print('\nWorking on model: %s' % (model))

    # loop over regions
    for region in regionlist:
        print('%s over region: %s' % (' '*10, region))

        # assign lower and upper latitude bounds for the latitudinal mean
        lat_lo = lat_bounds[region]['lo']
        lat_up = lat_bounds[region]['up']

        # loop over seasons
        for seas in seaslist:
            print('%s on season: %s' % (' '*20, seas))

            # create title string
            titlestr = make_titlestr(varname='Surface (black) and 500 hPa (red)', seas=seas, region=region)
            print(titlestr)

            datadir1 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (model, varname1, region, lat_lo, lat_up)
            datadir2 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (model, varname2, region, lat_lo, lat_up)
            plotdir = '/project2/tas1/miyawaki/projects/000_hotdays/plots/%s/comp_%s_%s/%s_%g_%g' % (model, varname1, varname2, region, lat_lo, lat_up)

            # create directories is they don't exist
            if not os.path.isdir(plotdir):
                os.makedirs(plotdir)

            # location of pickled percentile dmse data
            dcmse_file1 = '%s/dc%s_%s_%s.pickle' % (datadir1, varname1, region, seas)
            dcmse_file2 = '%s/dc%s_%s_%s.pickle' % (datadir2, varname2, region, seas)

            # load pickled data
            if mmm:
                dcmse_l_agg1 = np.empty([len(prct), len(modellist)])
                dcmse_o_agg1 = np.empty([len(prct), len(modellist)])
                dctas_l_agg1 = np.empty([len(prct), len(modellist)])
                dctas_o_agg1 = np.empty([len(prct), len(modellist)])
                dchus_l_agg1 = np.empty([len(prct), len(modellist)])
                dchus_o_agg1 = np.empty([len(prct), len(modellist)])
                dcmse_l_agg2 = np.empty([len(prct), len(modellist)])
                dcmse_o_agg2 = np.empty([len(prct), len(modellist)])
                dctas_l_agg2 = np.empty([len(prct), len(modellist)])
                dctas_o_agg2 = np.empty([len(prct), len(modellist)])
                dchus_l_agg2 = np.empty([len(prct), len(modellist)])
                dchus_o_agg2 = np.empty([len(prct), len(modellist)])
                for imodel in tqdm(range(len(modellist))):
                    currentmodel = modellist[imodel]

                    datadir1 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (currentmodel, varname1, region, lat_lo, lat_up)
                    datadir2 = '/project2/tas1/miyawaki/projects/000_hotdays/data/proc/%s/%s/%s_%g_%g' % (currentmodel, varname2, region, lat_lo, lat_up)

                    dcmse_file1 = '%s/dc%s_%s_%s.pickle' % (datadir1, varname1, region, seas)
                    dcmse_file2 = '%s/dc%s_%s_%s.pickle' % (datadir2, varname2, region, seas)

                    [dcmse_l_agg1[:,imodel], dcmse_o_agg1[:,imodel], dctas_l_agg1[:,imodel], dctas_o_agg1[:,imodel], dchus_l_agg1[:,imodel], dchus_o_agg1[:,imodel]] = pickle.load(open(dcmse_file1, 'rb'))
                    [dcmse_l_agg2[:,imodel], dcmse_o_agg2[:,imodel], dctas_l_agg2[:,imodel], dctas_o_agg2[:,imodel], dchus_l_agg2[:,imodel], dchus_o_agg2[:,imodel]] = pickle.load(open(dcmse_file2, 'rb'))

                # take multimodel mean
                dcmse_l1 = np.mean(dcmse_l_agg1,1)
                dcmse_o1 = np.mean(dcmse_o_agg1,1)
                dctas_l1 = np.mean(dctas_l_agg1,1)
                dctas_o1 = np.mean(dctas_o_agg1,1)
                dchus_l1 = np.mean(dchus_l_agg1,1)
                dchus_o1 = np.mean(dchus_o_agg1,1)

                dcmse_l2 = np.mean(dcmse_l_agg2,1)
                dcmse_o2 = np.mean(dcmse_o_agg2,1)
                dctas_l2 = np.mean(dctas_l_agg2,1)
                dctas_o2 = np.mean(dctas_o_agg2,1)
                dchus_l2 = np.mean(dchus_l_agg2,1)
                dchus_o2 = np.mean(dchus_o_agg2,1)

            else:
                [dcmse_l1, dcmse_o1, dctas_l1, dctas_o1, dchus_l1, dchus_o1] = pickle.load(open(dcmse_file1, 'rb'))
                [dcmse_l2, dcmse_o2, dctas_l2, dctas_o2, dchus_l2, dchus_o2] = pickle.load(open(dcmse_file2, 'rb'))

            ###########################
            # plot dcmse
            ###########################
            if seas is None:
                plotname = '%s/dcmse' % (plotdir)
            else:
                plotname = '%s/dcmse.%s' % (plotdir, seas)
            fig, ax = plt.subplots()
            vmin = 0
            vmax = 100
            ax.axhline(1,0, 100, linewidth=0.5, color='k')
            ax.plot(prct, dcmse_l2/dcmse_o2, ':r')
            ax.plot(prct, dctas_l2/dctas_o2, '-r')
            ax.plot(prct, dchus_l2/dchus_o2, '--r')
            ax.plot(prct, dcmse_l1/dcmse_o1, ':k', label='MSE')
            ax.plot(prct, dctas_l1/dctas_o1, '-k', label='Temperature')
            ax.plot(prct, dchus_l1/dchus_o1, '--k', label='Specific humidity')
            ax.set_title(titlestr)
            # make_title_sim_time(ax, sim, model=model, timemean=timemean)
            ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
            # if 'ymonmean' in timemean:
            #     ax.set_xticks(np.arange(0,12,1))
            #     ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
            # else:
            ax.set_xlim(0,100)
            ax.set_xlabel('Percentile')
            ax.set_ylabel('$\delta$ Land / $\delta$ Ocean (unitless)')
            # ax.set_yticks(np.arange(-90,91,30))
            # ax.xaxis.set_minor_locator(MultipleLocator(10))
            # ax.yaxis.set_minor_locator(MultipleLocator(10))
            plt.legend()
            fig.set_size_inches(5, 4)
            plt.savefig('%s.pdf' % (plotname), format='pdf', dpi=300)
            plt.close()

        # end seas loop

    # end region loop

    # stop after one loop if multimodel mean
    if mmm:
        break

# end model loop
