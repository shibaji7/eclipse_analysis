#!/usr/bin/env python

"""utils.py: utility module to support other functions."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import os
import ephem

import h5py
import numpy as np
from scipy import array
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
import glob
import pandas as pd
from pathlib import Path

def get_gridded_parameters(
    q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=True
):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[[xparam, yparam, zparam]]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    plotParamDF = plotParamDF.groupby([xparam, yparam]).mean().reset_index()
    plotParamDF = plotParamDF[[xparam, yparam, zparam]].pivot(xparam, yparam)
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y = np.meshgrid(x, y)
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
        np.isnan(plotParamDF[zparam].values), plotParamDF[zparam].values
    )
    return X, Y, Z

def get_folder(rad, beam, date, base="figures/rt/"):
    """
    Get folder by date
    """
    import os
    fold = os.path.join(base, date.strftime("%Y-%m-%d"), rad, "%02d"%beam)
    os.makedirs(fold, exist_ok=True)
    return fold

def read_params_2D(fname="pyrt/cfg/rt2D.json"):
    import json
    from types import SimpleNamespace

    with open(fname, "r") as f:
        param = json.load(f, object_hook=lambda x: SimpleNamespace(**x))
    return param

def clean():
    files = glob.glob(str(Path.home() / "matlab_crash_dump*"))
    for f in files:
        if os.path.exists(f): os.remove(f)
    return

def extrap1d(x, y, kind="linear"):
    """This method is used to extrapolate 1D paramteres"""
    interpolator = interp1d(x, y, kind=kind)
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(list(map(pointwise, array(xs))))

    return ufunclike

def interpolate_by_altitude(h, hx, param, scale="log", kind="cubic", method="intp"):
    if scale == "linear":
        pnew = (
            interp1d(h, param, kind=kind)(hx)
            if method == "intp"
            else extrap1d(h, param, kind=kind)(hx)
        )
    if scale == "log":
        pnew = (
            10 ** interp1d(h, np.log10(param), kind=kind)(hx)
            if method == "intp"
            else 10 ** extrap1d(h, np.log10(param), kind=kind)(hx)
        )
    return pnew

def smooth(x,window_len=11,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y


def read_eclispe_path(year=2024):
    fname = f"database/eclipse_path/{year}.csv"
    LatC, LonC = [], []
    LatN, LonN = [], []
    LatS, LonS = [], []
    with open(fname, "r") as f: lines = f.readlines()
    for line in lines:
        line = line.split("  ")
        locN, loc, locS = line[1], line[3], line[2]

        latcomp = -1 if "S" in loc else 1
        loc = loc.split(" ")
        LatC.append(
            latcomp*float(loc[0])+
            float(loc[1].\
                replace(".","").\
                replace("N","").\
                replace("S",""))/1e3
        )
        LonC.append(
            -1*float(loc[2])+
            float(loc[3].\
                replace(".","").\
                replace("W",""))/1e3
        )

        locS = locS.split(" ")
        LatS.append(
            latcomp*float(locS[0])+
            float(locS[1].\
                replace(".","").\
                replace("N","").\
                replace("S",""))/1e3
        )
        LonS.append(
            -1*float(locS[2])+
            float(locS[3].\
                replace(".","").\
                replace("W",""))/1e3
        )

        locN = locN.split(" ")
        LatN.append(
            latcomp*float(locN[0])+
            float(locN[1].\
                replace(".","").\
                replace("N","").\
                replace("S",""))/1e3
        )
        LonN.append(
            -1*float(locN[2])+
            float(locN[3].\
                replace(".","").\
                replace("W",""))/1e3
        )
    LatC, LonC = smooth(np.array(LatC))+0.4, smooth(np.array(LonC))
    LatS, LonS = smooth(np.array(LatS))+0.4, smooth(np.array(LonS))
    LatN, LonN = smooth(np.array(LatN))+0.4, smooth(np.array(LonN))
    o = pd.DataFrame()
    o["LatC"], o["LonC"] = LatC, LonC
    o["LatS"], o["LonS"] = LatS, LonS
    o["LatN"], o["LonN"] = LatN, LonN
    return o

class Eclipse(object):
    def __init__(self):
        return    

    def intersection(slef, r0, r1, d, n_s=100):
        A1 = np.zeros([n_s, n_s])
        A2 = np.zeros([n_s, n_s])
        I = np.zeros([n_s, n_s])
        x = np.linspace(-2.0*r0, 2.0*r0, num=n_s)
        y = np.linspace(-2.0*r0, 2.0*r0, num=n_s)
        xx, yy = np.meshgrid(x,y)
        A1[np.sqrt((xx+d)**2.0+yy**2.0) < r0] = 1.0
        n_sun = np.sum(A1)
        A2[np.sqrt(xx**2.0+yy**2.0) < r1] = 1.0
        S = A1 + A2
        I[S > 1] = 1.0
        eclipse = np.sum(I) / n_sun
        return(eclipse)

    def create_eclipse_shadow(self, d, lat, lon, alt):
        obs = ephem.Observer()
        t0 = ephem.date(
            (
                d.year,
                d.month,
                d.day,
                d.hour,
                d.minute,
                d.second,
            )
        )
        obs.lon, obs.lat = '%1.2f'%(lon), '%1.2f'%(lat) # ESR
        obs.elevation=alt
        obs.date= t0
        sun, moon = ephem.Sun(), ephem.Moon()
        sun.compute(obs)
        moon.compute(obs)
        r_sun=(sun.size/2.0)/3600.0
        r_moon=(moon.size/2.0)/3600.0
        s=np.degrees(ephem.separation((sun.az, sun.alt), (moon.az, moon.alt)))
        percent_eclipse=0.0
                
        if s < (r_moon+r_sun):
            if s < 1e-3: percent_eclipse=1.0
            else: percent_eclipse=self.intersection(r_moon,r_sun,s,n_s=100)
        if np.degrees(sun.alt) <= r_sun:
            if np.degrees(sun.alt) <= -r_sun: percent_eclipse=np.nan
            else:
                percent_eclipse = 1.0-((np.degrees(sun.alt)+r_sun)/(2.0*r_sun))*(1.0-percent_eclipse)
        return percent_eclipse