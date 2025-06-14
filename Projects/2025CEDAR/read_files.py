import xarray as xr
from loguru import logger
import pandas as pd
import datetime as dt
import pydarn
import numpy as np

from scipy.interpolate import RectBivariateSpline

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
import sys
sys.path.append("py/")
from plot import RangeTimePlot


RedBlackBlue=LinearSegmentedColormap.from_list(
    "RedBlackBlue",
    [
        (0.0, "#FF0000"),  # reddish
        (0.5, "#000000"),  # black
        (1.0, "#131DE3E8"),  # black
    ],
)

def read_mask(
    rad, beam,
    folder="../../raw_datasets/mask/mask/*.nc", 
    date_lim=[dt.datetime(2024,4,8,18), dt.datetime(2024,4,8,22)],
    mask="193",
):
    import glob
    files = glob.glob(folder)
    files.sort()
    dates = []
    hdw = pydarn.read_hdw_file(rad)
    latFull, lonFull = pydarn.Coords.GEOGRAPHIC(hdw.stid, center=True)
    nglat, nglon = latFull[:90, beam], lonFull[:90, beam]
    dhs = []
    for f in files:
        date = dt.datetime.strptime(
            f.split("/")[-1].split("_")[0], "%Y%m%d%H%M%S"
        )
        if (date>=date_lim[0]) & (date<=date_lim[1]):
            dates.append(date)
            logger.info(f"File: {f}")
            ds = xr.load_dataset(f)
            is_of = RectBivariateSpline(ds["glat"].values, ds["glon"].values, ds[mask].values)
            of = [is_of(ngt, ngl)[0,0] for ngt, ngl in zip(nglat, nglon)]
            dhs.append(of)
    dhs = np.array(dhs)
    dhs[dhs<=0.01] = np.nan
    range = np.arange(90)*45 + 180
    return (dates, range, dhs)

def create_TEC_map(
    sTEC, rad, beam,
    date_lim=[dt.datetime(2024,4,8,18), dt.datetime(2024,4,8,22)],
):
    dates = []
    hdw = pydarn.read_hdw_file(rad)
    latFull, lonFull = pydarn.Coords.GEOGRAPHIC(hdw.stid, center=True)
    nglat, nglon = latFull[:90, beam], lonFull[:90, beam]
    dhs = []
    for i, date in enumerate(sTEC["time"]):
        if (date>=date_lim[0]) & (date<=date_lim[1]):
            dates.append(date)
            logger.info(f"File: {date}")
            el = []
            for ngt, ngl in zip(nglat, nglon):
                j, k = np.argmin(np.abs(sTEC["glat"]-ngt)), np.argmin(np.abs(sTEC["glon"]-ngl))
                el.append(sTEC["tec"][j, k, i])
            dhs.append(el)
    range = np.arange(90)*45 + 180 
    dhs = np.array(dhs)  
    return (dates, range, dhs)

def load_WAL_nc_files(
    folder="../../raw_datasets/", 
    files=[
        "20240408.wal.a.despeck.fitacf3.nc",
        "20240408.wal.b.despeck.fitacf3.nc",
        "20240408.wal.c.despeck.fitacf3.nc",
        "20240408.wal.d.despeck.fitacf3.nc",
        "20240408.wal.e.despeck.fitacf3.nc"
    ], 
    epoch = dt.datetime(1858, 11, 17, tzinfo=dt.timezone.utc)
):
    D = []
    for file in files:
        fname = folder + file
        logger.info(f"File: {fname}")
        ds = xr.load_dataset(fname)
        df = ds.to_dataframe()
        df.mjd = df.mjd.apply(lambda x: x + epoch)
        D.append(df)
    D = pd.concat(D)
    D.rename(columns=dict(beam="bmnum", mjd="time"), inplace=True)
    D.tfreq /=1e3
    return D

def load_los_sTEC(
    folder="../../raw_datasets/",
    fname="conv_from_los_20240408_350km_30el.nc",
):
    file = folder + fname
    logger.info(f"File: {file}")
    ds = xr.load_dataset(file)
    sTEC = dict(
        time=pd.to_datetime(ds["time"].values),
        glon=ds["glon"].values,
        glat=ds["glat"].values,
        tec=ds["tec"].values,
    )
    return sTEC

def create_RTI(df, rad, beam, sTEC):
    fname = f"{rad}_{beam}.png"
    rti = RangeTimePlot(
        [0,2000], 
        [dt.datetime(2024,4,8,18), dt.datetime(2024,4,8,22)], 
        "", 
        1,
        font="sans-sarif",
    )
    ax = rti.addParamPlot(
        rad, df, 
        beam, title="",
        p_max=10, p_min=-10,
        xlabel="Time, UT", ylabel=r"Slant Range, km", 
        zparam="v", label=r"Velocity, $ms^{-1}$",
        cmap=RedBlackBlue, cbar=True, add_gflg=False,
        yparam="range", kind="scatter"
    )
    o = df[(df.bmnum==beam)]
    ax.text(
        0.05, 1.05, 
        f"Rad: {rad} / Beam: {beam} / $f_0=${o.tfreq.iloc[-1]} MHz / 8 April 2024, despeck [a-e]",
        ha="left", va="center", transform=ax.transAxes
    )
    # Overlay shadow:
    dates, range, of = read_mask(rad, beam)
    X, Y = np.meshgrid(dates, range)
    C = ax.contour(
        X, Y, 
        1-of.T, cmap="gray_r", zorder=2, alpha=1,
        levels=[0.15, 0.3, 0.45, 0.6, 0.75,]
    )
    ax.clabel(C, inline=True, fontsize=8)
    # Overlay TEC
    dates, range, el = create_TEC_map(sTEC, rad, beam)
    X, Y = np.meshgrid(dates, range)
    im = ax.scatter(
        X.ravel(), Y.ravel(), c=el.T.ravel(), s=10, marker="s", cmap="Greens",
        vmax=30, vmin=15, alpha=0.4
    )
    rti._add_colorbar(im, ax, "Greens", label="el, TECu", dx=0.2)
    ax.set_ylim(180, 2000)
    rti.save(fname)
    rti.close()
    return

if __name__ == "__main__":
    D = load_WAL_nc_files()
    sTEC = load_los_sTEC()
    create_RTI(D, "wal", 0, sTEC)
    create_RTI(D, "wal", 7, sTEC)
    create_RTI(D, "wal", 15, sTEC)
    create_RTI(D, "wal", 21, sTEC)
    