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
        
sys.path.append("py/")
from fan import Fan
from read_fitacf import Radar
from plot import RangeTimePlot
import eutils as utils

def setup():
    import os
    os.makedirs("figures_2021_Special", exist_ok=True)
    return


def generate_fov_overview(
    rads, date, beams=[15, 11, 7, 3], 
    cb=True, central_longitude=120.0, central_latitude=-45.0,
    extent=[-180, 180, -90, -50], plt_lats = np.arange(-90, -40, 10)
):
    fan = Fan(
        rads, date, f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, sup_title=False, figsize=(4, 3.5), dpi=1000
    )
    ax = fan.add_axes()
    for rad, col in zip(rads, ["r", "b"]):
        fan.overlay_fovs(
            rad,beams=beams if rad=="fir" else [],ax=ax, col=col, 
            maxGate=100 if rad=="fir" else 75,
        )
    ax.overlay_eclipse(True)
    fan.save(f"figures_2021_Special/fov.{date.strftime('%Y%m%d%H%M')}.png")
    fan.close() 
    return


def generate_conjugate_fov_overview(
    rads, conj_radar, date, beams=[15, 11, 7, 3], cb=True, 
    central_longitude=120.0, central_latitude=-45.0,
    extent=[-180, 180, -90, -50], plt_lats = np.arange(-90, -40, 10),
    overlay_eclipse_other_hemi=False, hemi="south", 
    other_instruments=[],
):
    fan = Fan(
        rads, date, f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, sup_title=False, figsize=(4, 3.5), dpi=1000
    )
    ax = fan.add_axes()
    for rad, col in zip(rads, ["r", "b"]):
        fan.overlay_fovs(
            rad,beams=beams if rad=="fir" else [],ax=ax, col=col, 
            maxGate=100 if rad=="fir" else 75,
        )
    ax.overlay_eclipse(True)

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
            if o_inst[1] == "isr":
                ax.overlay_instument(
                    o_inst[0], o_inst[2], o_inst[3], 
                    marker="o", markerColor="b", font_color="b",
                    xOffset=-5,yOffset=-1.5,
                )
            else:
                ax.overlay_instument(o_inst[0], o_inst[2], o_inst[3])
    fan.save(f"figures_2021_Special/fov.{hemi}.{date.strftime('%Y%m%d%H%M')}.png")
    fan.close() 
    return


def create_fan_plots(
    rads, dates, tfreq=None, channel=None, cb=False,
    central_longitude=80.0, central_latitude=-60.0,
    extent=[60, 130, -90, -45], plt_lats = np.arange(-90, -45, 10),
    overlay_eclipse_other_hemi=False,
    tags = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)"],
    p_max=500, p_min=200, mark_lon=120, yOffset=5, xOffset=-5,
):
    radars = dict()
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    for rad in rads:
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        if channel:
            df = df[df.channel==channel]
        df["unique_tfreq"] = df.tfreq#.apply(lambda x: int(x/0.5)*0.5)
        if tfreq: 
            df = df[df.tfreq.isin(tfreq)]
        v, tf = np.array(df.v), np.array(df.unique_tfreq)
        v[tf==10.5] *= -1
        df.v = v
        radar.df = df
        radars[rad] = radar
    
    fan = Fan(
        rads, dates[0], f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=3,sup_title=False,
        mark_lon=mark_lon
    )
    for j, date in enumerate(dates):
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5, #transform=fan.proj
        )
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5, #transform=fan.proj
        )
        print(">>>>>>>>>>>>>>>>>>>>>",XYZ.shape)
        for rad in rads:
            o = radars[rad].df.copy()
            o = o[
                (o.time>=date)
                & (o.time<=date+dt.timedelta(minutes=1))
            ]
            # o = o[o.bmnum==7]
            fan.generate_fov(
                rad, o, ax=ax, cbar=j==2,
                eclipse_cb=j==len(dates)-1, 
                # eclipse_cb=True,
                p_max=p_max, p_min=p_min,
                xOffset=5, yOffset=-1.5, 
                maxGate=100 if rad=="fir" else 75,
            )
        ax.text(0.05, 1.05, tags[j] + f" {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        # ax.add_square_grid(-60,-85,10)
        if j==0:
            ax.text(
                -0.05, 0.05, "Ch: [a, b]",
                ha="left", va="bottom",
                transform=ax.transAxes, fontsize="xx-small",
                rotation=90
            )
            ax.text(
                0.95, 1.05, f"$f_0$= {tfreq if tfreq else 'all'} MHz",
                ha="right", va="bottom",
                transform=ax.transAxes, fontsize="xx-small",
            )
        # apex = Apex(date)

        ## map from other hemisphere
        # lats, lons = np.arange(45, 90, 1), np.arange(-150, -70, 1)
        # lats, lons = np.meshgrid(lats, lons)
        # newglat, newglon, _ = apex.map_to_height(lats, lons, 100, 100, conjugate=True)
        # print(newglat.max(), newglat.min(), newglon)
        # p = utils.get_fov_eclipse(date, newglat, newglon)
        # xyz = ax.projection.transform_points(cartopy.crs.PlateCarree(), lons, lats)
        # x, y = xyz[:, :, 0], xyz[:, :, 1]
        # im = ax.contourf(
        #     x.T, y.T,
        #     p.T,
        #     cmap="Blues", alpha=0.6,
        #     levels=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
        #     transform=ax.projection
        # )
    fan.fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fan.save(f"figures_2021_Special/{date.strftime('%Y%m%d%H%M')}.png")
    fan.close()
    return

def create_mix_ts():
    from readmix import get_pot_drop
    from plot import RangeTimePlot
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter

    Phi0, time = get_pot_drop()

    rti = RangeTimePlot(
        100, [], 
        # r"MIX $\phi$, in kV during Dec 4, 2021 Eclipse", 
        "",
        num_subplots=1
    )
    ax = rti._add_axis()

    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    hours = mdates.HourLocator(byhour=range(0, 24, 6))
    ax.xaxis.set_major_locator(hours)
    ax.set_xlabel("Time, UT", fontdict={"size":12})
    ax.set_ylabel(r"$\Phi_0$, kV", fontdict={"size":12})
    # ax.set_ylabel(r"$J_p$, $\mu A/m^2$", fontdict={"size":12})
    ax.set_xlim(dt.datetime(2021, 12, 4), dt.datetime(2021, 12, 5))
    ax.plot(time, Phi0, "ko", ms=1)
    rti.save("figures_2021_Special/rti_mix.png")
    rti.close()
    print(">>>>>>>>>>>>>>>>")
    return

def create_rays():
    import rays
    rtos = [
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7, 30), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8, 15), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8, 30), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 9), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 9, 30), "fir", 0, limit_elvs=[20, 40]),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 9, 45), "fir", 0, limit_elvs=[20, 40]),
    ]
    rp = rays.PlotRays(rtos[0], nrows=len(rtos), ncols=1, arc=True)
    rp.lay_rays(
        text="(A) 7:00 UT",
    )
    for i, ray in enumerate(rtos[1:]):
        rp.lay_rays(
            rto=ray,
            text=f"({chr(ord('B')+i)}) {ray.event.strftime('%H:%M UT')}",
            add_cbar=False,
            add_tag=False,
        )
    rp.save(f"figures_2021_Special/Rays.png")
    rp.close()
    return


def create_fan_plots_stack(
    rads, dates, tfreq=[None], channel=[None], cb=False,
    central_longitude=80.0, central_latitude=-60.0,
    extent=[60, 130, -90, -45], plt_lats = np.arange(-90, -45, 10),
    tags = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)", "(K)", "(L)"],
    p_max=[500], p_min=[200], mark_lon=120, YO=[2, -3], XO=[-5, -1],
    labels = ["Velocity(fir), m/s", "Velocity(mcm), m/s"],
    colors= ["r", "b"], cmaps=[None, "Blues"]
):
    radars = dict()
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    for rad, tf0, ch in zip(rads, tfreq, channel):
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        if ch:
            df = df[df.channel==ch]
        df["unique_tfreq"] = df.tfreq#.apply(lambda x: int(x/0.5)*0.5)
        if tf0: 
            df = df[df.tfreq.isin(tf0)]
        v, tf = np.array(df.v), np.array(df.unique_tfreq)
        v[tf==10.5] *= -1
        df.v = v
        radar.df = df
        radars[rad] = radar
    
    fan = Fan(
        rads, dates[0], f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=4,sup_title=False,
        mark_lon=mark_lon
    )
    for j, date in enumerate(dates):
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5, #transform=fan.proj
        )
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5, #transform=fan.proj
        )
        for i, rad in enumerate(rads):
            o = radars[rad].df.copy()
            o = o[
                (o.time>=date)
                & (o.time<=date+dt.timedelta(minutes=1))
            ]
            # o = o[o.bmnum==7]
            fan.generate_fov(
                rad, o, ax=ax, cbar=((j==3) * (i==0))|((j==7) * (i==1)),
                eclipse_cb=j==len(dates)-1, 
                # eclipse_cb=True,
                p_max=p_max[i], p_min=p_min[i],
                xOffset=XO[i], yOffset=YO[i], 
                maxGate=100 if rad=="fir" else 75,
                label=labels[i], col=colors[i],
                cmap=cmaps[i],
            )
            if j==0:
                ax.text(
                    -0.05, 0.05, "Ch[fir]: [1, 2]",
                    ha="left", va="bottom",
                    transform=ax.transAxes, fontsize="xx-small",
                    rotation=90
                )
                ax.text(
                    0.95, 1.05, f"$f_0$[fir]= {tfreq[0] if tfreq[0] else 'all'} MHz",
                    ha="right", va="bottom",
                    transform=ax.transAxes, fontsize="xx-small",
                )
            if j==4:
                ax.text(
                    -0.05, 0.05, "Ch[mcm]: [1]",
                    ha="left", va="bottom",
                    transform=ax.transAxes, fontsize="xx-small",
                    rotation=90
                )
                ax.text(
                    0.95, 1.01, f"$f_0$[mcm]= all MHz",
                    ha="right", va="bottom",
                    transform=ax.transAxes, fontsize="xx-small",
                )
        ax.text(0.05, 1.05, tags[j] + f" {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        # # ax.add_square_grid(-60,-85,10)
        
    fan.fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fan.save(f"figures_2021_Special/{date.strftime('%Y%m%d%H%M')}.png")
    fan.close()
    return

def create_map_plots(
    extent=[-180, 180, -90, -50], 
    plt_lats = np.arange(-90, -49, 10), cb=False, mark_lon=-50,
    central_longitude=80, central_latitude=-70.0,
):
    from readmix import get_2D_data, get_sd_data
    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=4, sup_title=False,
        mark_lon=mark_lon
    )
    dates = [
        dt.datetime(2021, 12, 4, 6),
        dt.datetime(2021, 12, 4, 6, 30),
        dt.datetime(2021, 12, 4, 7, 0),
        dt.datetime(2021, 12, 4, 7, 30),
        dt.datetime(2021, 12, 4, 7, 40),
        dt.datetime(2021, 12, 4, 7, 50),
        dt.datetime(2021, 12, 4, 8),
        dt.datetime(2021, 12, 4, 8, 14),
        dt.datetime(2021, 12, 4, 8, 30),
        dt.datetime(2021, 12, 4, 8, 44),
        dt.datetime(2021, 12, 4, 9),
        dt.datetime(2021, 12, 4, 9, 30)
    ]
    for j, date in enumerate(dates):
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        ax.overlay_eclipse(j==len(dates)-1)
        ax.text(0.05, 1.05, f"({chr(ord('A')+j)}) {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        if j==0:
            ax.text(
                -0.05, 0.05, "",
                ha="left", va="bottom",
                transform=ax.transAxes, fontsize="xx-small",
                rotation=90
            )
            ax.text(
                0.95, 1.05, "",
                ha="right", va="bottom",
                transform=ax.transAxes, fontsize="xx-small",
            )
        data, lats, lons = get_2D_data(date, var="J_par")
        XYZ = fan.proj.transform_points(
            fan.geo, 
            lons, 
            lats
        )
        data = np.ma.masked_where(data==0, data)
        im = ax.contourf(
            XYZ[:, :, 0], XYZ[:, :, 1], data,
            cmap="jet_r", alpha=0.8, levels=np.arange(-1.2, 1.2, 0.6),
            extend="both"
        )
        if j==3:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$J_{||}$, $\mu A/m^2$")

        data, lats, lons = get_sd_data(date)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            lons, 
            lats
        )
        data = np.ma.masked_where(data==0., data)
        im = ax.contourf(
            XYZ[:, :, 0], XYZ[:, :, 1], data,
            cmap="Spectral",
            alpha=0.5,
            levels=np.arange(-45, 46, 15),
        )
        if j==7:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$\Phi_0$, kV")


    fan.fig.subplots_adjust(hspace=0.1, wspace=0.02)
    fan.save(f"figures_2021_Special/Maps.png")
    fan.close()
    return