# This script contains a set of utilities for the CESM2-SF (single forcing) runs

import sys
sys.path.append('/home/miyawaki/scripts/common/CASanalysis/CASutils')
sys.path.append('/glade/u/home/miyawaki/scripts/common/CASanalysis/CASutils')
from lensread_utils import lens2memnamegen_first50,lens2memnamegen_second50

def emem(fo):
    # DESCRIPTION
    # Returns a list of ensemble member numbers applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
    
    # OUTPUT 
    # lmem  : LIST of ensemble member numbers for that forcing

    # list of ensemble member numbers
    if fo=='lens2':
        lmem=['%03d' % (e) for e in range(1,100+1)] 

    return lmem


def conf(fo,isim):
    # DESCRIPTION
    # Returns the configuration name applicable for the type of run

    if yr<=2014:
        if fo=='lens2':
            if isim<51:
                cnf='BHISTcmip6'
            else:
                cnf='BHISTsmbb'
    else:
        if fo=='lens2':
            if isim<51:
                cnf='BSSP370cmip6'
            else:
                cnf='BSSP370smbb'

    return cnf

def simu(fo):
    # DESCRIPTION
    # Returns the simulation name applicable for the type of run

    if fo=='lens2':
        sim1=['LE2-%s' % (e) for e in lens2memnamegen_first50(50)]
        sim2=['LE2-%s' % (e) for e in lens2memnamegen_second50(50)]
        sim=[*sim1,*sim2]

    return sim

def sely(fo,cl):
    # DESCRIPTION
    # Returns the simulation years applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
    # cl    : STRING describing the climatology (fut=future, his=historical)
    
    # OUTPUT 
    # sim   : STRING of simulation name

    # list of ensemble member numbers

    if fo=='lens2':
        lyr=['19500101-19591231', '19600101-19691231', '19700101-19791231','19800101-19891231','19900101-19991231','20000101-20091231','20100101-20141231','20150101-20241231','20250101-20341231', '20350101-20441231', '20450101-20541231','20550101-20641231','20650101-20741231','20750101-20841231', '20850101-20941231', '20950101-21001231']

    return lyr

def casename(fo,isim):
    return 'b.e21.%s.f09_g17.%s'%(conf(fo,isim),simu(fo))

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

def rename_vn(varn):
    d={
        'trefht':   'tas',
        'fsm':      'snm',
        'fsno':     'snc',
            }
    try:
        return d[varn]
    except:
        return varn
