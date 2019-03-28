import os,sys


def myreadfile(fnm, mode):
    if '.gz' == fnm[-3:]:
        fnm = fnm[:-3]
    if os.path.isfile(fnm):
        f = open(fnm, mode)
    elif os.path.isfile(fnm+'.gz'):
        import gzip
        f = gzip.open(fnm+'.gz', mode)
    else:
        print 'file: {} or its zip file does NOT exist'.format(fnm)
        sys.exit(1)
    return f

def checkfilegz(name):
    if os.path.isfile(name):
        return name
    elif os.path.isfile(name+'.gz'):
        return name+'.gz'
    else:
        return None

def loadedgelist(ifile):
    '''
    load edge list from file
    format: src det value1, value2......
    '''
    edgelist = []
    with myreadfile(ifile, 'rb') as fin:
        for line in fin:
            coords = line.strip()
            edgelist.append(coords)
    return edgelist
