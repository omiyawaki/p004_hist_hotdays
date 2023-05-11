def make_titlestr(**kwargs):
    varname = kwargs.get('varname')
    seas = kwargs.get('seas')
    region = kwargs.get('region')

    levelstr = get_levelstr(varname)

    if seas is None:
        if region == 'trop':
            titlestr = 'Tropics, %s' % (levelstr)
        elif region == 'nhmid':
            titlestr = 'NH Midlatitudes, %s' % (levelstr)
        elif region == 'shmid':
            titlestr = 'SH Midlatitudes, %s' % (levelstr)
    else:
        if region == 'trop':
            titlestr = 'Tropics, %s, %s' % (levelstr, seas.upper())
        elif region == 'nhmid':
            titlestr = 'NH Midlatitudes, %s, %s' % (levelstr, seas.upper())
        elif region == 'shmid':
            titlestr = 'SH Midlatitudes, %s, %s' % (levelstr, seas.upper())

    return titlestr

def get_levelstr(varname):

    if varname == 'tas':
        levelstr = '2 m Temperature'
    elif varname == 'mses':
        levelstr = 'Surface'
    elif 'ta' in varname:
        levelstr = '%.0f hPa Temperature' % (1e-2*float(varname[2:]))
    elif 'mse' in varname:
        levelstr = '%.0f hPa' % (1e-2*float(varname[3:]))
    else:
        levelstr = varname

    return levelstr
