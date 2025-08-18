from xarray import open_dataset
import datetime as dt

import sys
sys.path.append("py/")
from fan import Fan
import numpy as np
import eutils as utils

import cartopy

import scienceplots
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]


def create_2021_tec(
):
    central_longitude=80.0
    central_latitude=-60.0
    extent=[60, 130, -90, -45]
    extent=[-180, 180, -90, -60]
    plt_lats = np.arange(-90, -45, 10)
    mark_lon=120

    dates = [dt.datetime(2021, 12, 4, 6, 45) + dt.timedelta(minutes=15*i) for i in range(9)]
    datas = read_datasets(dates)
    fan = Fan(
        [], dates[0], f"", cb=False,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=3,sup_title=False,
        mark_lon=mark_lon
    )
    for i, date in enumerate(dates):
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=i==0, add_time=False)
        data = datas[i]
        Lat, Lon = np.meshgrid(data["lat"], data["lon"])
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            Lon, Lat
        )
        im = ax.pcolor(
            xyz[:, :, 0],
            xyz[:, :, 1],
            data["data"].T,
            cmap="Spectral",
            vmax=0.25, vmin=-0.25,
            transform=ax.projection,
            shading='auto',
            zorder=3,
            alpha=0.6,
        )
        if i==len(dates)-1:
            pos = ax.get_position()
            mpos = [0.025, 0.0125, 0.015, 0.5]
            cpos = [
                pos.x1 + mpos[0],
                pos.y0 + mpos[1],
                mpos[2],
                pos.height * mpos[3],
            ]
            cax = fan.fig.add_axes(cpos)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            cb.set_label(r"dTEC, TECu")
        ax.text(0.05, 0.05, f"({chr(65+i)}) {date.strftime('%H%M UT')}", ha="left", va="center", transform=ax.transAxes)


        Lat, Lon = np.meshgrid(np.arange(-90, 50, 0.5), np.arange(-180, 180, 1))
        p = utils.get_fov_eclipse(
            date, Lat, Lon
        )
        p[p<=0] = np.nan
        p[p>1] = np.nan
        xyz = ax.projection.transform_points(
            cartopy.crs.PlateCarree(),
            Lon, Lat
        )
        cf = ax.contourf(
            xyz[:, :, 0],
            xyz[:, :, 1],
            p,
            levels=np.arange(0., 1.01, 0.25),
            cmap="Blues",
            transform=ax.projection,
            extend="max",
            zorder=1,
        )
        if i==2:
            pos = ax.get_position()
            mpos = [0.025, 0.0125, 0.015, 0.5]
            cpos = [
                pos.x1 + mpos[0],
                pos.y0 + mpos[1],
                mpos[2],
                pos.height * mpos[3],
            ]
            cax = fan.fig.add_axes(cpos)
            cb = fan.fig.colorbar(cf, ax=ax, cax=cax)
            cb.set_label(r"Obscuration ($\mathcal{O}$)")

    fan.fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fan.save(f"figures_2021_Special/tec_{date.strftime('%Y%m%d%H%M')}.png")
    fan.close()
    return

def read_datasets(dates):
    dtec = []
    for d in dates:
        ds = open_dataset(f"database/December2021/{d.strftime('%Y%m%d%H')}_dtec.nc")
        i = int(d.minute/5)
        dtec.append(dict(
            time=d,
            data=ds.dtec.values[:, :, i],
            lat=ds.lat.values,
            lon=ds.lon.values,
        ))
        ds.close()
    return dtec

if __name__ =="__main__":
    create_2021_tec()