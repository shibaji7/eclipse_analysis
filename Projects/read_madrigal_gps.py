import time
import h5py
import numpy

def getSiteData(madrigalFile, siteArr, site):
    with h5py.File(madrigalFile, 'r') as f:
        # get a list of all the indices with the right site
        site_indices = siteArr == site
        siteData = f['Data']['Table Layout'][site_indices]
        return(siteData)

def getTimeData(madrigalFile, timeArr, unixTime):
    with h5py.File(madrigalFile, 'r') as f:
        time_indices = timeArr == unixTime
        timeData = f['Data']['Table Layout'][time_indices]
        return(timeData)

def getTimeGpsData(madrigalFile, timeArr, satTypeArr, unixTime):
    with h5py.File(madrigalFile, 'r') as f:
        indices = numpy.logical_and(timeArr == unixTime, satTypeArr == b'GPS')
        timeData = f['Data']['Table Layout'][indices]
        return(timeData)

