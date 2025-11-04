
import sys
sys.path.extend([
    "py/", 
    "Projects/2021_Eclipse_Special/", 
    "Projects/2021_Eclipse_Special/ASHLEY_v1/ASHLEY-A/",
    "Projects/2021_Eclipse_Special/ASHLEY_v1/ASHLEY-E/",
])

import sys
import numpy as np
from numpy import concatenate, ones, shape, transpose, vstack
import cartopy
from apexpy import Apex
import pydarn
import datetime as dt
from loguru import logger
import bz2
from pydarn import RangeEstimation
import pandas as pd
from matplotlib.colors import LogNorm
        
sys.path.append("py/")
from fan import Fan
from read_fitacf import Radar
from plot import RangeTimePlot
import eutils as utils

def get_1D_density_data(lat, lon, h=150, ext="euv",):
    data = {}
    from xarray import open_dataset
    indx = "001" if ext=="base" else "002"
    file = f"/home/chakras4/Modeled-Electron-Densities/GITM/20211204/result_202112_{ext}-{indx}.nc"
    ds = open_dataset(file)
    dates = [
        dt.datetime(y, m, d, h, mn) 
        for y, m, d, h, mn in zip(
            ds.year.values[:], ds.month.values[:], ds.day.values[:], 
            ds.hour.values[:], ds.minute.values[:]
            )
        ]
    glon = np.mod(360 + lon, 360)
    data["date"] = dates
    data["glat"], lon["glon"] = lat, glon
    j = np.argmin(np.abs((ds.alt.values[:]/1000)-h))
    k, l = np.argmin(np.abs((ds.alt.values[:]/1000)-h))
    ds.close()
    return data

def get_2D_density_data(date, ext="euv", lat_bound=[-90, -50], lon_bound=[-180, 180], h=150):
    data = {}
    from xarray import open_dataset
    indx = "001" if ext=="base" else "002"
    file = f"/home/chakras4/Modeled-Electron-Densities/GITM/20211204/result_202112_{ext}-{indx}.nc"
    ds = open_dataset(file)
    dates = [
        dt.datetime(y, m, d, h, mn) 
        for y, m, d, h, mn in zip(
            ds.year.values[:], ds.month.values[:], ds.day.values[:], 
            ds.hour.values[:], ds.minute.values[:]
            )
        ]
    data["date"] = date
    data["glat"], data["glon"], data["alt"] = (
        ds.glat.values[:],
        ds.glon.values[:],
        ds.alt.values[:]
    )
    j = np.argmin(np.abs((ds.alt.values[:]/1000)-h))
    i = dates.index(date)
    data["ne"] = ds.dene.values[i, j, :, :]
    ds.close()
    return data

def create_simultaneous_plots(
    extent=[-180, 180, -90, -50], 
    plt_lats = np.arange(-90, -49, 10), 
    cb=False, mark_lon=-50,
    central_longitude=90, central_latitude=-70.0,
    kind="euv"
):
    dates = [
        dt.datetime(2021, 12, 4, 7),
        dt.datetime(2021, 12, 4, 7,15),
        dt.datetime(2021, 12, 4, 7,30),
        dt.datetime(2021, 12, 4, 7,45),
        dt.datetime(2021, 12, 4, 8),
        dt.datetime(2021, 12, 4, 8,15),
    ]

    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=2, ncols=3, sup_title=False,
        mark_lon=mark_lon, coord="geo",
        figsize=(3, 3.2)
    )
    for j, d in enumerate(dates):
        fan.date = d
        ax = fan.add_axes(add_coords=j==0, add_time=True)
        data = get_2D_density_data(d, ext=kind)
        glat, glon = data["glat"], data["glon"]
        glat, glon = np.meshgrid(glat, glon)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            glon, glat
            
        )
        data["ne"] = np.ma.masked_where(data["ne"]==0, data["ne"])
        im = ax.pcolor(
            XYZ[:, :, 0], XYZ[:, :, 1], data["ne"].T,
            alpha=0.5, zorder=2, cmap="Spectral_r",
            norm=LogNorm(vmin=1e10, vmax=5e11)
        )
        ax.overlay_fov("fir", lineColor="m", maxGate=110,)
        ax.overlay_radar("fir", font_color="m", yOffset=1, xOffset=3, markerColor="m", fontSize=8)
        if kind=="euv":
            ax.overlay_eclipse(j==len(dates)-1)
        if j==2:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$N_e$, $/cm$")
    fan.save(f"figures_2021_Special/overlay_ne_{kind}.png")
    fan.fig.savefig(f"figures_2021_Special/overlay_ne_{kind}.png", dpi=100, bbox_inches="tight")
    fan.close()
    return

create_simultaneous_plots()
create_simultaneous_plots(kind="base")
