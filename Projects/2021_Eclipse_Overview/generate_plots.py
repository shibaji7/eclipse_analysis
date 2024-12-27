"""
    This python module is used to plot the dataset and geometry.
"""

import sys
import numpy as np
from numpy import concatenate, ones, shape, transpose, vstack
import cartopy
from apexpy import Apex
import pydarn
        
sys.path.append("py/")
from fan import Fan
from read_fitacf import Radar
from plot import RangeTimePlot
import eutils as utils

def setup():
    import os
    os.makedirs("figures", exist_ok=True)
    return

def generate_fov_overview(
    rads, date, beams=[15, 11, 7, 3, 0], 
    cb=True, central_longitude=120.0, central_latitude=-45.0,
    extent=[-180, 180, -90, -50], plt_lats = np.arange(-90, -40, 10)
):
    fan = Fan(
        rads, date, f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats
    )
    ax = fan.add_axes()
    for rad in rads:
        fan.overlay_fovs(rad,beams=beams,ax=ax)
    fan.save(f"figures/fanbeam.{date.strftime('%Y%m%d%H%M')}.png")
    fan.close() 
    return

def generate_conjugate_fov_overview(
    rads, conj_radar, date, beams=[], cb=True, 
    central_longitude=120.0, central_latitude=-45.0,
    extent=[-180, 180, -90, -50], plt_lats = np.arange(-90, -40, 10),
    overlay_eclipse_other_hemi=False, hemi="south", 
    other_instruments=[],
):
    fan = Fan(
        rads, date, f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats
    )
    ax = fan.add_axes()
    for rad in rads:
        fan.overlay_fovs(rad,beams=beams,ax=ax)

    apex = Apex(date)
    for col, rad in zip(["r", "b", "g"], conj_radar):
        hdw = pydarn.read_hdw_file(rad)
        fov = pydarn.Coords.GEOGRAPHIC(hdw.stid)
        newglat, newglon, _ = apex.map_to_height(fov[0], fov[1], 100, 100, conjugate=True)
        lat, lon, _ = apex.map_to_height([hdw.geographic.lat], [hdw.geographic.lon], 100, 100, conjugate=True)
        
        sgate, egate = 0, 70
        sbeam, ebeam = 0, hdw.beams
        xyz = fan.proj.transform_points(cartopy.crs.PlateCarree(), newglon.T, newglat.T)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate(
            (
                x[sbeam, sgate:egate],
                x[sbeam:ebeam, egate],
                x[ebeam, egate:sgate:-1],
                x[ebeam:sbeam:-1, sgate],
            )
        )
        contour_y = concatenate(
            (
                y[sbeam, sgate:egate],
                y[sbeam:ebeam, egate],
                y[ebeam, egate:sgate:-1],
                y[ebeam:sbeam:-1, sgate],
            )
        )
        ax.plot(
            contour_x,
            contour_y,
            color=col,
            zorder=2,
            linewidth=0.6,
            ls="--",
            alpha=0.6,
        )
        ax.scatter(
            lon,
            lat,
            s=2,
            marker="o",
            color=col,
            zorder=2,
            transform=cartopy.crs.PlateCarree(),
            lw=0.8,
            alpha=0.4,
        )
        lat, lon = lat - 1.5, lon + 5
        ax.text(
                lon,
                lat,
                rad.upper(),
                ha="center",
                va="center",
                transform=cartopy.crs.PlateCarree(),
                fontdict={"color": col, "size": "xx-small"},
                alpha=0.8,
            )
    
    if overlay_eclipse_other_hemi:
        year = date.year
        o = utils.read_eclispe_path(year)
        keys, colors = ["C", "N", "S"], ["k", "r", "r"]
        for k, c in zip(keys, colors):
            newglat, newglon, _ = apex.map_to_height(o["Lat"+k].tolist(), o["Lon"+k].tolist(), 100, 100, conjugate=True)
            xy = fan.proj.transform_points(cartopy.crs.PlateCarree(), newglon.T, newglat.T)
            x, y = xy[:, 0], xy[:, 1]
            ax.plot(
                x, y,
                color=c,
                zorder=2,
                linewidth=0.8,
                ls="--",
            )

        # # Overlay eclipse shadow
        # alts = np.array([100])
        # lats = np.linspace(-90, 0, num=361)
        # lons = np.linspace(-180, 180, num=361)
        # p, _, _= utils.get_eclipse(date, alts, lats, lons)
        # p = np.ma.masked_invalid(p)[0,0,:,:]
        # obs = np.copy(p)
        # obs[obs>1.] = np.nan
        # newglat, newglon, _ = apex.map_to_height(
        #     lats, lons, 100, 100, 
        #     conjugate=True
        # )
        # im = ax.contourf(
        #     newglon,
        #     newglat,
        #     obs,
        #     transform=cartopy.crs.PlateCarree(),
        #     cmap="Blues", alpha=0.6,
        #     levels=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        # )
    if len(other_instruments):
        for o_inst in other_instruments:
            print(o_inst)
            ax.overlay_instument(o_inst[0], o_inst[2], o_inst[3])
    fan.save(f"figures/fanbeam.{hemi}.{date.strftime('%Y%m%d%H%M')}.png")
    fan.close() 
    return

def create_rti_plots(
    rad, dates, beam=15, yscale="Gn", 
    range=[0,4500], channel=None, tfreq=12.
):
    date = dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    title = fr"Rad: {rad} / Beam: {beam} / Date:  {date}"
    radar = Radar(rad, dates, type="fitacf")
    radar.calculate_ground_range()
    df = radar.df.copy()
    if channel:
        df = df[df.channel==channel]
    df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
    if tfreq: 
        df = df[df.unique_tfreq==tfreq]
    rti = RangeTimePlot(
        range, 
        dates, 
        title, 
        2,
        font="sans-sarif",
    )
    ax = rti.addParamPlot(
        rad, df, 
        beam, "", 
        p_max=30, p_min=-30,
        xlabel="Time, UT", ylabel="Slant Range, km", 
        zparam="v", label=r"Velocity, $ms^{-1}$",
        cmap="Spectral", cbar=True, add_gflg=False,
        yparam="srange", kind="scatter"
    )
    rti.overlay_eclipse_shadow(rad, beam, dates, ax, True)
    ax.set_ylim(range)
    ax.set_xlim(dates)
    ax = rti.addParamPlot(
        rad, df, 
        beam, "", 
        p_max=30, p_min=-30,
        xlabel="Time, UT", ylabel="Maped Ground Range, km", 
        zparam="v", label=r"Velocity, $ms^{-1}$",
        cmap="Spectral", cbar=True, add_gflg=False,
        yparam="Gn", kind="scatter"
    )
    ax.set_ylim(range[0]/2, range[1]/2)
    ax.set_xlim(dates)
    rti.save(f"figures/rti.{rad}-{beam}.png")
    rti.close()
    return