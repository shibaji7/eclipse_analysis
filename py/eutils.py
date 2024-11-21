import pandas as pd
import ephem
import numpy as np
import datetime as dt

def smooth(x,window_len=11,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = np.ones(window_len,"d")
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

def magnetic_inclination_angle(lats, lons, date):
    import pyIGRF
    FACT = 180.0 / np.pi
    r = 6371.0
    I = np.zeros_like(lats)*np.nan
    for i in range(lats.shape[0]):
        for j in range(lats.shape[1]):
            _, ix, _, _, _, _, _ = pyIGRF.igrf_value(
                lats[i, j], lons[i, j], r, date.year
            )
            I[i,j] = ix
    return I

def setsize(size=12):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import mplstyle
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Tahoma",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
    ]
    mpl.rcParams.update(
        {"xtick.labelsize": size, "ytick.labelsize": size, "font.size": size}
    )
    return

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.round(plotParamDF[xparam], 2)
        plotParamDF[yparam] = np.round(plotParamDF[yparam], 2)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z

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
            if np.degrees(sun.alt) <= -r_sun: percent_eclipse=2
            else:
                percent_eclipse = 1.0-((np.degrees(sun.alt)+r_sun)/(2.0*r_sun))*(1.0-percent_eclipse)
        return percent_eclipse

def get_rti_eclipse(
    dates, lats, lons, alt=300
):
    from tqdm import tqdm
    e = Eclipse()
    p = np.nan * np.zeros((len(dates), len(lats)))
    for i, d in enumerate(tqdm(dates)):
        for j, lat, lon in zip(range(len(lats)), lats, lons):
            p[i,j] = e.create_eclipse_shadow(d, lat, lon, alt)
    return p

def get_eclipse(
    date, alts=np.array([300]),
    lats=np.linspace(0,90,num=90*2),
    lons=np.linspace(-180,180,num=91*2),
    n_t=1, dtx=60.0,
):
    n_alts=len(alts)
    n_lats=len(lats)
    n_lons=len(lons)
    e = Eclipse()
    
    p=np.zeros([n_t,n_alts,n_lats,n_lons])
    times=np.arange(n_t)*dtx
    dts=[]
    for ti,t in enumerate(times):
        d = date + dt.timedelta(seconds=t)
        #print("Time %1.2f (s)"%(t))
        for ai,alt in enumerate(alts):
            for lai,lat in enumerate(lats):
                for loi,lon in enumerate(lons):
                    p[ti,ai,lai,loi] = e.create_eclipse_shadow(d, lat, lon, alt)
        dts.append(d)
    return (p,times,dts)

def create_movie(folder, outfile, pat, fps=3):
    """
    Create movies from pngs
    """
    import cv2
    import glob
    files = glob.glob(f"{folder}/{pat}")
    files.sort()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    img = cv2.imread(files[0])
    height, width, layers = img.shape
    size = (width, height)
    out = cv2.VideoWriter(f"{folder}/{outfile}", fourcc, fps, size)
    for idx in range(len(files)):
        img = cv2.imread(files[idx])
        out.write(img)
    out.release()
    return