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
    if len(other_instruments):
        for o_inst in other_instruments:
            ax.overlay_instument(o_inst[0], o_inst[2], o_inst[3])
    fan.save(f"figures/fanbeam.{hemi}.{date.strftime('%Y%m%d%H%M')}.png")
    fan.close() 
    return

def create_rti_plots(
    rad_beams, dates, beam=15, yscale="Gn", 
    range=[0,2500], channel=None, tfreq=None
):

    from supermag import SuperMAG
    sm = SuperMAG.FetchSM(
        "database/", 
        dates, 
        uid="shibaji7", 
        stations=["fcc", "brd", "ott"]
    )
    print(sm.sm_data.head())
    date = dates[1].strftime("%d %b, %Y") if dates[0].day == dates[1].day else dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    rti = RangeTimePlot(
        range, 
        dates, 
        date, 
        2,
        font="sans-sarif",
    )
    for rad_beam in rad_beams:
        rad, beam = rad_beam[0], rad_beam[1]
        title = fr"Rad: {rad} / Beam: {beam}"
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        if channel:
            df = df[df.channel==channel]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        if tfreq: 
            df = df[df.unique_tfreq==tfreq]
        ax = rti.addParamPlot(
            rad, df, 
            beam, title=title,
            p_max=30, p_min=-30,
            xlabel="Time, UT", ylabel="Mapped GS Range, km", 
            zparam="v", label=r"Velocity, $ms^{-1}$",
            cmap="jet_r", cbar=True, add_gflg=False,
            yparam="Chisham_gsmap", kind="scatter"
        )
        # rti.overlay_eclipse_shadow(rad, beam, dates, ax, True)
        rti.add_conjugate_eclipse(rad, beam, dates, ax)
        ax.set_ylim(range)
        ax.set_xlim(dates)
        o = sm.sm_data.copy()
        o = o[o.iaga=="FCC"]
        rti.add_supermag_TS(ax, o, "dB [FCC], nT", )
    rti.save(f"figures/rti.{rad}-{beam}.png")
    rti.close()
    return