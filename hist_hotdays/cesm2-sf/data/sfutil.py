# This script contains a set of utilities for the CESM2-SF (single forcing) runs

import sys
sys.path.append('/home/miyawaki/scripts/common/CASanalysis/CASutils')
sys.path.append('/glade/u/home/miyawaki/scripts/common/CASanalysis/CASutils')
from lensread_utils import lens2memnamegen_first50

def emem(fo):
    # DESCRIPTION
    # Returns a list of ensemble member numbers applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
    
    # OUTPUT 
    # lmem  : LIST of ensemble member numbers for that forcing

    # list of ensemble member numbers
    if fo=='lens':
        lmem=['%03d' % (e) for e in range(1,50+1)] 
    elif fo=='ee':
        lmem=['%03d' % (e) for e in range(101,115+1)] 
    elif fo=='xaaer':
        lmem=['%03d' % (e) for e in range(1,3+1)] 
    else:
        lmem=['%03d' % (e) for e in range(1,15+1)] 

    return lmem


def conf(fo,cl):
    # DESCRIPTION
    # Returns the configuration name applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
    # cl    : STRING describing the climatology (fut=future, his=historical)
    
    # OUTPUT 
    # cnf   : STRING of configuration name

    # list of ensemble member numbers
    if cl=='fut':
        if fo=='lens':
            cnf='BSSP370cmip6'
            # cnf='BSSP370smbb'
        elif fo=='xaaer':
            cnf='BSSP370cmip6'
        else:
            cnf='B1850cmip6'

    elif cl=='his':
        if fo=='lens':
            cnf='BHISTcmip6'
            # cnf='BHISTsmbb'
        elif fo=='xaaer':
            cnf='BHISTcmip6'
        else:
            cnf='B1850cmip6'

    return cnf

def simu(fo,cl):
    # DESCRIPTION
    # Returns the simulation name applicable for the type of run

    # INPUT
    # fo    : STRING describing the forcing (ghg=greenhouse gases, aaer=anthropogenic aerosols, bmb=biomass burning, ee=everything else, xaaer=all forcing except anthropogenic aerosols)
    # cl    : STRING describing the climatology (fut=future, his=historical)
    
    # OUTPUT 
    # sim   : STRING of simulation name or
    #         LIST of string of simulation names (for LENS only)

    # list of ensemble member numbers

    if fo=='lens':
        sim=['LE2-%s' % (e) for e in lens2memnamegen_first50(50)]
    elif fo=='xaaer':
        sim='CESM2-SF-xAER'
    else:
        if cl=='fut':
            sim='CESM2-SF-%s-SSP370' % (fo.upper())
        elif cl=='his':
            sim='CESM2-SF-%s' % (fo.upper())

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

    if cl=='fut':
        if fo=='lens':
            lyr=['20250101-20341231', '20350101-20441231', '20450101-20541231']
        else:
            lyr=['20250101-20341231', '20350101-20441231', '20450101-20501231'] # future
    elif cl=='his':
            lyr=['19200101-19291231', '19300101-19391231', '19400101-19491231'] # past

    return lyr
