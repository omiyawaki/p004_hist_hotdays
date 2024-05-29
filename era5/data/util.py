def rename_vn(varn):
    d={
        'T2m':      'tas',
            }
    try:
        return d[varn]
    except:
        return varn
