# This script contains a set of utilities for the CESM2-SF (single forcing) runs

import sys
import xarray as xr
# sys.path.append('/home/miyawaki/scripts/common/CASanalysis/CASutils')

def mods(fo):
    # DESCRIPTION
    # Returns a list of model names applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (e.g., ssp245)
    
    # OUTPUT 
    # lmd  : LIST of model names for that forcing

    if fo in ['historical','ssp245','ssp370']:

        # ALL AVAILABLE SSP370 ZG, TAS and STF
        lmd=sorted(['ACCESS-CM2','CanESM5','CESM2', 'IITM-ESM', 'MIROC-ES2L','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM','NorESM2-MM','UKESM1-0-LL','EC-Earth3', 'KACE-1-0-G', 'MIROC6','MRI-ESM2-0','IPSL-CM6A-LR','TaiESM1'])
        # lmd=sorted(['ACCESS-CM2', 'ACCESS-ESM1-5','CanESM5','CESM2', 'IITM-ESM', 'MIROC-ES2L','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM','NorESM2-MM','UKESM1-0-LL','EC-Earth3', 'KACE-1-0-G', 'MIROC6','MRI-ESM2-0','IPSL-CM6A-LR','TaiESM1'])
        # lmd=sorted(['ACCESS-CM2', 'ACCESS-ESM1-5','CanESM5','CESM2', 'IITM-ESM','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM','NorESM2-MM','UKESM1-0-LL','EC-Earth3', 'KACE-1-0-G', 'MIROC6','MRI-ESM2-0'])
        # lmd=sorted(['IITM-ESM'])
        # lmd=sorted(['MIROC-ES2L'])
        # lmd=['MPI-ESM1-2-HR','EC-Earth3','MRI-ESM2-0'] # HISTO
        # lmd=['MPI-ESM1-2-HR','EC-Earth3','MRI-ESM2-0'] # SSP

        # ALL AVAILABLE SSP370 TAS and STF
        # lmd=['ACCESS-CM2', 'ACCESS-ESM1-5','CanESM5','CESM2', 'IITM-ESM', 'MIROC-ES2L','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM','NorESM2-MM','UKESM1-0-LL','CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6','MRI-ESM2-0']
        # lmd=sorted(['ACCESS-CM2', 'ACCESS-ESM1-5','CanESM5','CESM2', 'IITM-ESM', 'MIROC-ES2L','MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'NorESM2-LM','NorESM2-MM','UKESM1-0-LL','CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'KACE-1-0-G', 'MIROC6','MRI-ESM2-0'])

        # BYRNE LIST
        # lmd=['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'GFDL-CM4', 'GFDL-ESM4', 'HadGEM3-GC31-LL', 'INM-CM4-8', 'INM-CM5-0', 'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']

    return lmd

def emem(md):
    # DESCRIPTION
    # Returns the ensemble name applicable for the type of run

    # INPUT
    # md    : STRING describing the model (e.g., ACCESS-CM1)
    
    # OUTPUT 
    # mem   : STRING of ensemble name or

    if md in ['CNRM-CM6-1','CNRM-CM6-1-HR','CNRM-ESM2-1','MIROC-ES2L']:
        mem='r1i1p1f2'
    elif md in ['HadGEM3-GC31-LL']:
        mem='r1i1p1f3'
    elif md in ['CESM2']:
        mem='r11i1p1f1'
    elif md in ['CAMS-CSM1-0']:
        mem='r2i1p1f1'
    elif md in ['UKESM1-0-LL']:
        mem='r9i1p1f2'
    else:
        mem='r1i1p1f1'

    return mem

def simu(fo,cl,yr):
    # DESCRIPTION
    # Returns the simulation name applicable for the type of run 

    # INPUT
    # fo    : STRING describing the forcing (e.g., ssp245)
    # cl    : STRING describing the climatology (fut=future, his=historical)
    # cl    : FLOAT describing the year (fut=future, his=historical)
    
    # OUTPUT 
    # sim   : STRING of simulation name
    
    if cl=='his':
        sim='historical'
    elif cl=='fut':
        sim=fo
    elif cl=='tseries':
        if yr<2015:
            sim='historical'
        else:
            sim=fo

    return sim


def year(cl,md,byr):
    # DESCRIPTION
    # Returns a list of years applicable for the type of run

    # INPUT
    # cl    : STRING describing the climatology (fut=future, his=historical)
    # md    : STRING describing the model (e.g., ACCESS-CM1)
    # byr   : LIST of floats describing the bounds of time of interest (e.g., [2030,2050])
    
    # OUTPUT 
    # lyr  : LIST of years for that forcing

    if cl=='fut':
        if byr[0]==2030 and byr[1]==2050:
            # one file
            if md in ['CanESM5','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM6A-LR','KACE-1-0-G']:
                lyr=['20150101-21001231']
            # 50 year inc
            elif md in ['ACCESS-CM2','ACCESS-ESM1-5','INM-CM4-8','INM-CM5-0','MRI-ESM2-0']:
                lyr=['20150101-20641231']
            # 50 year inc (non-relative)
            elif md in ['HadGEM3-GC31-LL','UKESM1-0-LL']:
                lyr=['20150101-20491230','20500101-21001230']
            # 25 year inc
            elif md in ['BCC-CSM2-MR']:
                lyr=['20150101-20391231','20400101-20641231']
            # 20 year inc
            elif md in ['MPI-ESM1-2-LR']:
                lyr=['20150101-20341231','20350101-20541231']
            # 10 year inc
            elif md in ['CESM2-WACCM','TaiESM1']:
                lyr=['20250101-20341231','20350101-20441231','20450101-20541231']
            # 10 year inc (non-relative)
            elif md in ['NorESM2-LM']:
                lyr=['20210101-20301231','20310101-20401231','20410101-20501231']
            # 1 year inc
            elif md in ['AWI-CM-1-1-MR','MIROC-ES2L']:
                lyr=['%04d0101-%04d1231' % (yr,yr) for yr in range(byr[0],byr[1]+1)]

        elif byr[0]==2080 and byr[1]==2100:
            # one file
            if md in ['CanESM5','CNRM-CM6-1','CNRM-ESM2-1','IPSL-CM6A-LR','KACE-1-0-G']:
                lyr=['20150101-21001231']
            # one file (360 day)
            if md in ['KACE-1-0-G']:
                lyr=['20150101-21001230']
            # 50 year inc
            elif md in ['ACCESS-CM2','ACCESS-ESM1-5','INM-CM4-8','INM-CM5-0','MRI-ESM2-0']:
                lyr=['20650101-21001231']
            # 50 year inc (non-relative)
            elif md in ['HadGEM3-GC31-LL','UKESM1-0-LL']:
                lyr=['20500101-21001230']
            # 25 year inc
            elif md in ['BCC-CSM2-MR']:
                lyr=['20650101-20891231','20900101-21001231']
            # 20 year inc
            elif md in ['MPI-ESM1-2-LR']:
                lyr=['20750101-20941231','20950101-21001231']
            # 10 year inc
            elif md in ['CESM2-WACCM','TaiESM1']:
                lyr=['20750101-20841231','20850101-20941231','20950101-21001231']
            # 10 year inc (non-relative)
            elif md in ['NorESM2-LM']:
                lyr=['20710101-20801231','20810101-20901231','20910101-21001231']
            # 1 year inc
            elif md in ['AWI-CM-1-1-MR','MIROC-ES2L']:
                lyr=['%04d0101-%04d1231' % (yr,yr) for yr in range(byr[0],byr[1]+1)]

    elif cl=='his':
        if byr[0]==1920 and byr[1]==1940:
            error('Must redownload original data')
        elif byr[0]==1980 and byr[1]==2000:
            lyr=['19700101-20141231']

    return lyr

def load_raw(odir,varn,byr,se):
    # load raw data
    if 'gwl' in byr:
        fn='%s/lm.%s_%s.%s.nc' % (odir,varn,byr,se)
    else:
        fn='%s/lm.%s_%g-%g.%s.nc' % (odir,varn,byr[0],byr[1],se)
    print('\n Loading data to composite...')
    ds = xr.open_dataset(fn)
    print('\n Done.')
    return ds

