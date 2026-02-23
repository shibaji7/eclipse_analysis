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
    # xyz = fan.proj.transform_points(cartopy.crs.PlateCarree(), [], newglat.T)
    ax.scatter(
        164.2255,-74.6243, 
        s=20,
        marker="^",
        color="k",
        zorder=2,
        transform=fan.geo,
        lw=0.8,
        alpha=0.8,
    )
    ax.text(164.24-2, -74.62+2, "Jang Bogo", transform=cartopy.crs.PlateCarree(), ha="center", va="bottom", fontdict={"color": "k", "size": "x-small"}, alpha=0.8)
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
    from readmix import get_pot_drop, parse_pot_data
    from plot import RangeTimePlot
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter

    for rad, tf0, ch in zip(["mcm"], [None], [None]):
        radar = Radar(rad, [dt.datetime(2021,12,4), dt.datetime(2021,12,5)], type="fitacf")
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

    Jpar, time = get_pot_drop()
    Phi0, _ = get_pot_drop(None, "Pot")

    rti = RangeTimePlot(
        100, [dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)], 
        # r"MIX $\phi$, in kV during Dec 4, 2021 Eclipse", 
        "",
        num_subplots=3
    )
    ax = rti.addParamPlot(
        "mcm", radar.df, 7, "04 December, 2021", 
        p_max=600, p_min=100,
        cmap="GnBu", cbar=True,
        xlabel="",
    )
    import eutils
    p = eutils.get_rti_eclipse(
        [dt.datetime(2021,12,4,5)+dt.timedelta(seconds=30*i) for i in range(5*60*2)],
        radar.get_lat_lon_along_beam(7)[0], radar.get_lat_lon_along_beam(7)[1]
    )
    cs = ax.contour(
        [dt.datetime(2021,12,4,5)+dt.timedelta(seconds=30*i) for i in range(5*60*2)],
        np.arange(76),
        p.T,
        colors="k", 
        linewidths=0.5,
        levels=[0.2, 0.4, 0.6, 0.75, 1.0],
        zorder=1, alpha=0.6,
    )
    ax.clabel(cs, inline=True, fontsize=6, fmt='%.2f')
    ax.axvline(dt.datetime(2021, 12, 4, 5, 29), ls="--", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7, 33), ls="-", lw=0.8, color="r")
    ax.axvline(dt.datetime(2021, 12, 4, 8, 6), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 9, 37), ls="--", lw=0.8, color="k")
    ax.text(0.05, 0.95, "(a) mcm/7", ha="left", va="center", transform=ax.transAxes)

    from read_digisonde import consolidate_horizontal_data
    dates, vhf, vhf_err, vhf_base, vhf_base_err, lat, lon, time_dig, eof = consolidate_horizontal_data()
    # o = parse_pot_data()
    ax = rti._add_axis()
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    hours = mdates.HourLocator(byhour=range(0, 24, 1))
    ax.xaxis.set_major_locator(hours)
    ax.set_ylabel(r"$\mathcal{V}_H$, $m/s$", fontdict={"size":12, "color":"b"})
    ax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
    ax.errorbar(dates, vhf, yerr=vhf_err*.15, color="b", fmt="o", ms=1, elinewidth=0.5)
    ax.errorbar(dates, vhf_base, yerr=vhf_base_err*.15, color="k", fmt="^", ms=1, elinewidth=0.5)
    ax.tick_params(axis="y", colors="b",)
    ax.spines["right"].set_color("b")
    ax.set_ylim(-100, 700)
    # ax.plot(o["time, UT"], o["pot_drop, kV"], "ko", ms=1)
    ax.axvline(dt.datetime(2021, 12, 4, 5, 29), ls="--", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7, 33), ls="-", lw=0.8, color="r")
    ax.axvline(dt.datetime(2021, 12, 4, 8, 6), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 9, 37), ls="--", lw=0.8, color="k")
    ax.text(0.05, 0.95, r"(b) Jang Bogo/$(\theta,\phi)=(74.6^{\circ}S, 164.2^{\circ}E)$", ha="left", va="center", transform=ax.transAxes)
    tax = ax.twinx()
    tax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    tax.xaxis.set_major_locator(hours)
    tax.set_ylabel(r"EOF / $\mathcal{O}$", fontdict={"size":12, "color":"r"})
    tax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
    tax.tick_params(axis="y", colors="r",)
    tax.spines["right"].set_color("r")
    tax.plot(time_dig, eof, "r-", lw=0.8)
    tax.set_ylim(0, 1.2)
    

    ax = rti._add_axis()
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    hours = mdates.HourLocator(byhour=range(0, 24, 1))
    ax.xaxis.set_major_locator(hours)
    ax.set_xlabel("Time, UT", fontdict={"size":12})
    ax.set_ylabel(r"$J_{||}$, $\mu A/m^2$", fontdict={"size":12})
    ax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
    ax.plot(time, Jpar, "ko", ms=1)
    ax.text(0.05, 0.95, "(c) AMPERE FACs, CPCP (pyMIX)", ha="left", va="center", transform=ax.transAxes)
    ax.axvline(dt.datetime(2021, 12, 4, 5, 29), ls="--", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 7, 33), ls="-", lw=0.8, color="r")
    ax.axvline(dt.datetime(2021, 12, 4, 8, 6), ls="-", lw=0.8, color="k")
    ax.axvline(dt.datetime(2021, 12, 4, 9, 37), ls="--", lw=0.8, color="k")
    ax.set_ylim(0, 4)
    ax = ax.twinx()
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    hours = mdates.HourLocator(byhour=range(0, 24, 1))
    ax.tick_params(axis="y", colors="r",)
    ax.spines["right"].set_color("r")
    ax.xaxis.set_major_locator(hours)
    ax.set_ylabel(r"$\Phi_0$, $kV$", fontdict={"size":12, "color":"r"})
    ax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
    ax.plot(time, Phi0, "ro", ms=1)
    ax.set_ylim(0, 80)
    rti.save("figures_2021_Special/rti_mix.png")
    rti.close()
    return

def create_rays(run_name="Dec2021_gitm_base_Modeled"):
    import rays
    rtos_base = [
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7, 30), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8, 15), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        
    ]

    run_name = "Dec2021_gitm_eclipse_Modeled" 
    rtos_ecl = [
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 7, 30), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
        rays.RayTraceObject(dt.datetime(2021, 12, 4, 8, 15), "fir", 7, limit_elvs=[15, 40], run_name=run_name),
    ]
    rp = rays.PlotRays(rtos_base[0], nrows=len(rtos_ecl), ncols=2, arc=False, lw=.8,)
    rp.lay_rays(
        text="(A) 7:00 UT",
        add_cbar=False,
        add_time=False,
    )
    ax_num = [2, 4, 6]
    for i, ray in enumerate(rtos_base[1:]):
        rp.axnum = ax_num[i]
        rp.lay_rays(
            rto=ray,
            text=f"({chr(ord('B')+i)}) {ray.event.strftime('%H:%M UT')}",
            add_cbar=False,
            add_tag=False,
            dtype="Base",
            add_time=False,
            xlabel="" if i!=2 else "Ground Range, km",
        )
    ax_num = [1, 3, 5, 7]
    for i, ray in enumerate(rtos_ecl):
        rp.axnum = ax_num[i]
        rp.lay_rays(
            rto=ray,
            text=f"({chr(ord('E')+i)}) {ray.event.strftime('%H:%M UT')}",
            add_cbar=i==0,
            add_tag=i==0,
            dtype="Eclipse",
            add_time=False,
            xlabel="" if i!=3 else "Ground Range, km",
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
    colors= ["r", "b"], cmaps=["jet_r", "GnBu"]
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
        plt_lats=plt_lats, nrows=3, ncols=2,sup_title=False,
        mark_lon=mark_lon, figsize=(3,3),
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
                rad, o, ax=ax, cbar=((j==3) * (i==0))|((j==1) * (i==1)),
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
                    transform=ax.transAxes, fontsize="x-small",
                    rotation=90
                )
                ax.text(
                    0.95, 1.05, f"$f_0$[fir]= {tfreq[0] if tfreq[0] else 'all'} MHz",
                    ha="right", va="bottom",
                    transform=ax.transAxes, fontsize="x-small",
                )
            if j==2:
                ax.text(
                    -0.05, 0.05, "Ch[mcm]: [1]",
                    ha="left", va="bottom",
                    transform=ax.transAxes, fontsize="x-small",
                    rotation=90
                )
                ax.text(
                    0.95, 1.01, f"$f_0$[mcm]= all MHz",
                    ha="right", va="bottom",
                    transform=ax.transAxes, fontsize="x-small",
                )
        ax.text(0.05, 1.05, tags[j] + f" {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "x-small", "weight": "bold", "color": "k"})
        # # ax.add_square_grid(-60,-85,10)
        
    fan.fig.subplots_adjust(hspace=0.1, wspace=0.1)
    fan.save(f"figures_2021_Special/{date.strftime('%Y%m%d%H%M')}.png")
    fan.close()
    return

def create_overlay_amp_plots(
    extent=[-180, 180, -90, -60], 
    plt_lats = np.arange(-90, -49, 10), cb=False, mark_lon=-50,
    central_longitude=80, central_latitude=-90.0,
):
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,10)]

    from readmix import get_2D_data, get_sd_data, get_imfs
    from matplotlib.colors import TwoSlopeNorm
    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=2, sup_title=False,
        mark_lon=mark_lon, coord="geo", figsize=(3, 3)
    )
    dates = [
        # dt.datetime(2021, 12, 4, 6),
        # dt.datetime(2021, 12, 4, 6, 30),
        dt.datetime(2021, 12, 4, 7, 0),
        dt.datetime(2021, 12, 4, 7, 30),
        dt.datetime(2021, 12, 4, 7, 40),
        dt.datetime(2021, 12, 4, 7, 50),
        dt.datetime(2021, 12, 4, 8),
        # dt.datetime(2021, 12, 4, 8, 14),
        # dt.datetime(2021, 12, 4, 8, 30),
        # dt.datetime(2021, 12, 4, 8, 44),
        dt.datetime(2021, 12, 4, 9),
        # dt.datetime(2021, 12, 4, 9, 30)
    ]
    for j, date in enumerate(dates):
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        

        pot, plats, plons = get_2D_data(date, var="Pot")
        Jpar, jlats, jlons = get_2D_data(date,)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            plons, 
            plats
        )
        pot = np.ma.masked_where(np.abs(pot)==0., pot)
        im = ax.pcolor(
            XYZ[:, :, 0], XYZ[:, :, 1], pot,
            alpha=0.8, norm=TwoSlopeNorm(vcenter=0, vmin=-20, vmax=20),
             zorder=2, cmap="RdBu"
        )
        if j==1:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$\Phi [GITM + AMPERE]$, $kV$")
        utils.setsize(12)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            jlons, 
            jlats
        )
        Jpar = np.ma.masked_where(np.abs(Jpar)==0., Jpar)
        im = ax.contourf(
            XYZ[:, :, 0], XYZ[:, :, 1], Jpar,
            alpha=0.7, norm=TwoSlopeNorm(vcenter=0, vmin=-1.5, vmax=1.5),
            zorder=3, cmap="jet", levels=np.arange(-2, 2.1, 0.25)
        )
        if j==3:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            cb.set_ticks([-2, -1, 0, 1, 2])
            utils.setsize(10)
            cb.set_label(r"$AMPERE$, $\mu A/m^2$")
        utils.setsize(12)
        imfs = get_imfs(date)

        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5,)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5)
        ax.scatter(
            164.24,-74.62, 
            s=20,
            marker="^",
            color="k",
            zorder=3,
            transform=cartopy.crs.PlateCarree(),
            lw=0.8,
            alpha=0.8,
        )
        from read_digisonde import get_hv_by_date
        hv = get_hv_by_date(date)
        print("Digisonde HV:", hv)
        q = ax.quiver(
            np.array([[164.24]]), 
            np.array([[-74.62]]),
            np.array([[hv["VXF"]]]), np.array([[hv["VYF"]]]), 
            transform=cartopy.crs.PlateCarree(),
            headwidth=2, headlength=2, scale=1500, color="m", 
            zorder=3
        )
        if j==0:
            qk = ax.quiverkey(
                q,
                X=0.2,
                Y=0.2,
                U=300,
                # angle=90,
                label="300 m/s",
                labelpos="S",
                coordinates="axes",
                labelsep=0.05
            )
            qk.text.set_fontsize("x-small")
            # qk.text.set_rotation(90)
        txt = fr"$\phi_0$={np.round(np.max(pot)-np.min(pot),1)} kV" + "\n"
        txt = txt + fr"$\theta$={np.round(imfs['IMF.tilt, deg'].iloc[0],1)}$^\circ$"+ "\n"
        txt = txt + fr"$|B|$={np.round(imfs['IMF.B, nT'].iloc[0],1)} nT"+ "\n"
        txt = txt + (r"$Vz_{jb}$=%.1f m/s"%(hv['VZF']))
        ax.text(0.05, 1.05, f"({chr(ord('A')+j)}) {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        ax.text(0.05, 0.95, txt, ha="left", va="top", transform=ax.transAxes, fontdict={"size": 6, "color": "k"})
        ax.overlay_eclipse(j==len(dates)-1)

    fan.fig.subplots_adjust(hspace=0.1, wspace=0.02)
    fan.save(f"figures_2021_Special/Maps.png")
    fan.close()
    return

def create_map_plots(
    extent=[-180, 180, -90, -50], 
    plt_lats = np.arange(-90, -49, 10), cb=False, mark_lon=-50,
    central_longitude=80, central_latitude=-70.0,
):
    # Ream MCM and DMSP
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,10)]
    radar = Radar("mcm", dates, type="fitacf")
    radar.calculate_ground_range()
    df = radar.df.copy()
    df["unique_tfreq"] = df.tfreq
    radar.df = df
    # Read others
    from readmix import get_2D_data, get_sd_data, get_imfs
    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=4, sup_title=False,
        mark_lon=mark_lon, coord="geo"
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
        o = radar.df.copy()
        o = o[
            (o.time>=date)
            & (o.time<=date+dt.timedelta(minutes=1))
        ]
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        ax.overlay_eclipse(j==len(dates)-1)

        fan.generate_fov(
            "mcm", o, ax=ax, cbar=(j==7),
            # eclipse_cb=j==len(dates)-1, 
            eclipse_cb=False,
            p_max=500, p_min=300,
            xOffset=-1, yOffset=-3, 
            maxGate=75,
            label="Velocity, m/s", col="b",
            cmap="GnBu",
        )

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
        data, lats, lons = get_2D_data(date, var="Pot")
        XYZ = fan.proj.transform_points(
            fan.geo, 
            lons, 
            lats
        )
        data = np.ma.masked_where(data==0, data)
        im = ax.pcolor(
            XYZ[:, :, 0], XYZ[:, :, 1], data,
            alpha=0.5, vmax=20, vmin=-20,
             zorder=2, cmap="RdBu"
        )
        
        if j==3:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$\Phi [pyMIX]$, $kV$")
        imfs = get_imfs(date)

        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5,)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5)
        ax.scatter(
            164.24,-74.62, 
            s=20,
            marker="^",
            color="k",
            zorder=2,
            transform=cartopy.crs.PlateCarree(),
            lw=0.8,
            alpha=0.8,
        )
        from read_digisonde import get_hv_by_date
        hv = get_hv_by_date(date)
        print("Digisonde HV:", hv)
        q = ax.quiver(
            np.array([[164.24]]), 
            np.array([[-74.62]]),
            np.array([[hv["VXF"]]]), np.array([[hv["VYF"]]]), 
            transform=cartopy.crs.PlateCarree(),
            headwidth=2, headlength=2, scale=1500, color="m", 
            zorder=3
        )
        if j==0:
            qk = ax.quiverkey(
                q,
                X=1.05,
                Y=0.8,
                U=500,
                angle=90,
                label="500 m/s",
                labelpos="E",
                coordinates="axes",
                labelsep=0.05
            )
            # Shrink and rotate the quiver key label for compact layout
            qk.text.set_fontsize("x-small")
            qk.text.set_rotation(90)

        # data, lats, lons = get_sd_data(date)
        # XYZ = fan.proj.transform_points(
        #     fan.geo, 
        #     lons, 
        #     lats
        # )
        # data = np.ma.masked_where(data==0., data)
        # im = ax.contourf(
        #     XYZ[:, :, 0], XYZ[:, :, 1], data,
        #     cmap="cool",
        #     alpha=0.8,
        #     # vmax=45, vmin=-45
        #     levels=np.arange(-45, 46, 15),
        # )
        txt = fr"$\phi_0$={np.round(np.max(data)-np.min(data),1)} kV" + "\n"
        txt = txt + fr"$\theta$={np.round(imfs['IMF.tilt, deg'].iloc[0],1)}$^\circ$"+ "\n"
        txt = txt + fr"$|B|$={np.round(imfs['IMF.B, nT'].iloc[0],1)} nT"+ "\n"
        # txt = txt + fr"n={np.round(imfs['nvecs'].iloc[0],1)}"+ "\n"
        txt = txt + (r"$Vz_{jb}$=%.1f m/s"%(hv['VZF']))
        ax.text(0.05, 1.05, f"({chr(ord('A')+j)}) {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        ax.text(0.05, 0.95, txt, ha="left", va="top", transform=ax.transAxes, fontdict={"size": 6, "color": "k"})
        # if j==7:
        #     utils.setsize(10)
        #     cpos = [1.05, 0.1, 0.025, 0.6]
        #     cax = ax.inset_axes(cpos, transform=ax.transAxes)
        #     cb = fan.fig.colorbar(im, ax=ax, cax=cax)
        #     utils.setsize(10)
        #     cb.set_label(r"$\Phi$ [TS18], kV")


    fan.fig.subplots_adjust(hspace=0.1, wspace=0.02)
    fan.save(f"figures_2021_Special/Maps.png")
    fan.close()
    return


def plot_hall_conductivity(extent=[-180, 180, -90, -50], 
    plt_lats = np.arange(-90, -49, 10), cb=False, mark_lon=-50,
    central_longitude=80, central_latitude=-70.0,
    cond="Pedersen",
):
    # Ream MCM and DMSP
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,10)]
    radar = Radar("mcm", dates, type="fitacf")
    radar.calculate_ground_range()
    df = radar.df.copy()
    df["unique_tfreq"] = df.tfreq
    radar.df = df
    # Read others
    from readmix import get_2D_data, get_sd_data, get_imfs
    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=4, sup_title=False,
        mark_lon=mark_lon, coord="geo"
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
        o = radar.df.copy()
        o = o[
            (o.time>=date)
            & (o.time<=date+dt.timedelta(minutes=1))
        ]
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        ax.overlay_eclipse(j==len(dates)-1)

        fan.generate_fov(
            "mcm", o, ax=ax, cbar=(j==7),
            # eclipse_cb=j==len(dates)-1, 
            eclipse_cb=False,
            p_max=500, p_min=300,
            xOffset=-1, yOffset=-3, 
            maxGate=75,
            label="Velocity, m/s", col="b",
            cmap="GnBu",
        )

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
        data, lats, lons = get_2D_data(date, var=cond)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            lons, 
            lats
        )
        data = np.ma.masked_where(data==0, data)
        im = ax.pcolor(
            XYZ[:, :, 0], XYZ[:, :, 1], data,
            alpha=0.5, vmax=10, vmin=0,
             zorder=2, cmap="Spectral_r"
        )
        
        if j==3:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$\Sigma_%s$, $Siemens$"%("H" if cond=="Hall" else "P"))
        imfs = get_imfs(date)

        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5,)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5)
        ax.scatter(
            164.24,-74.62, 
            s=20,
            marker="^",
            color="k",
            zorder=2,
            transform=cartopy.crs.PlateCarree(),
            lw=0.8,
            alpha=0.8,
        )
        from read_digisonde import get_hv_by_date
        hv = get_hv_by_date(date)
        print("Digisonde HV:", hv)
        q = ax.quiver(
            np.array([[164.24]]), 
            np.array([[-74.62]]),
            np.array([[hv["VXF"]]]), np.array([[hv["VYF"]]]), 
            transform=cartopy.crs.PlateCarree(),
            headwidth=2, headlength=2, scale=1500, color="m", 
            zorder=3
        )
        if j==0:
            qk = ax.quiverkey(
                q,
                X=1.05,
                Y=0.8,
                U=500,
                angle=90,
                label="500 m/s",
                labelpos="E",
                coordinates="axes",
                labelsep=0.05
            )
            # Shrink and rotate the quiver key label for compact layout
            qk.text.set_fontsize("x-small")
            qk.text.set_rotation(90)

        # data, lats, lons = get_sd_data(date)
        # XYZ = fan.proj.transform_points(
        #     fan.geo, 
        #     lons, 
        #     lats
        # )
        # data = np.ma.masked_where(data==0., data)
        # im = ax.contourf(
        #     XYZ[:, :, 0], XYZ[:, :, 1], data,
        #     cmap="cool",
        #     alpha=0.8,
        #     # vmax=45, vmin=-45
        #     levels=np.arange(-45, 46, 15),
        # )
        print(imfs)
        txt = fr"$\phi_0$={np.round(np.max(data)-np.min(data),1)} kV" + "\n"
        txt = txt + fr"$\theta$={np.round(imfs['IMF.tilt, deg'].iloc[0],1)}$^\circ$"+ "\n"
        txt = txt + fr"$|B|$={np.round(imfs['IMF.B, nT'].iloc[0],1)} nT"+ "\n"
        # txt = txt + fr"n={np.round(imfs['nvecs'].iloc[0],1)}"+ "\n"
        txt = txt + (r"$Vz_{jb}$=%.1f m/s"%(hv['VZF']))
        ax.text(0.05, 1.05, f"({chr(ord('A')+j)}) {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        ax.text(0.05, 0.95, txt, ha="left", va="top", transform=ax.transAxes, fontdict={"size": 6, "color": "k"})
        # if j==7:
        #     utils.setsize(10)
        #     cpos = [1.05, 0.1, 0.025, 0.6]
        #     cax = ax.inset_axes(cpos, transform=ax.transAxes)
        #     cb = fan.fig.colorbar(im, ax=ax, cax=cax)
        #     utils.setsize(10)
        #     cb.set_label(r"$\Phi$ [TS18], kV")


    fan.fig.subplots_adjust(hspace=0.1, wspace=0.02)
    fan.save(f"figures_2021_Special/Cond_{cond}_Maps.png")
    fan.close()
    return


def create_Digisonde_plots():
    from plot import RangeTimePlot
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    from read_digisonde import consolidate_data

    vxf, vyf, vzf = (
        consolidate_data("VXF"), 
        consolidate_data("VYF"), 
        consolidate_data("VZF")
    )

    rti = RangeTimePlot(
        100, [dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)], 
        # r"MIX $\phi$, in kV during Dec 4, 2021 Eclipse", 
        "",
        num_subplots=3
    )

    vhm = np.sqrt(vxf[1]**2 + vyf[1]**2)
    # vht = 
    vhf = (vxf[0], vhm, )
    ranges = [(-100, 600), (-300, 300), (-80, 80)]
    multipliers = [0.25, 1, 0.05]
    components = ["X", "Y", "Z"]
    for i, v in enumerate([vxf, vyf, vzf]):
        ax = rti._add_axis()
        ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        ax.set_ylabel(r"$\mathcal{V}^{jb}_{%s}$, $m/s$"%components[i], fontdict={"size":15, "color":"b"})
        ax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
        ax.errorbar(v[0], v[1], yerr=v[2]*multipliers[i], color="b", fmt="o", ms=1, elinewidth=0.5)
        ax.tick_params(axis="y", colors="b",)
        ax.spines["left"].set_color("b")
        ax.errorbar(v[0], v[3], yerr=v[4]*multipliers[i], color="k", fmt="^", ms=1, elinewidth=0.5)
        tax = ax.twinx()
        tax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
        tax.xaxis.set_major_locator(hours)
        tax.set_ylabel(r"EOF / $\mathcal{O}$", fontdict={"size":15, "color":"r"})
        tax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
        tax.plot(v[-2], v[-1], "r-", lw=0.8)
        tax.tick_params(axis="y", colors="r",)
        tax.spines["right"].set_color("r")
        tax.set_ylim(0, 1.05)
        ax.text(0.05, 0.95, f"({chr(i+97)})", ha="left", va="center", transform=ax.transAxes)
        ax.set_ylim(ranges[i])
        if i==0:
            ax.text(0.05, 1.05, r"Jang Bogo Dynasonde / $(\theta,\phi)= (74.6^{\circ} S, 164.2^{\circ} E)$", ha="left", va="center", transform=ax.transAxes, fontdict={"size":15})

    ax.set_xlabel("Time, UT", fontdict={"size":15})
    rti.save("figures_2021_Special/digisonde_summary.png")
    rti.close()
    return

def create_MCM_plots():
    from plot import RangeTimePlot
    import matplotlib.dates as mdates
    from matplotlib.dates import DateFormatter
    
    ddates = [
        [dt.datetime(2021,12,3), dt.datetime(2021,12,4)],
        [dt.datetime(2021,12,4), dt.datetime(2021,12,5)],
        [dt.datetime(2021,12,5), dt.datetime(2021,12,6)],
    ]
    drads = []
    for rad, tf0, ch, dates in zip(["mcm", "mcm", "mcm"], [None, None, None], [None, None, None], ddates):
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
        drads.append(radar)

    rti = RangeTimePlot(
        100, [dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)], 
        # r"MIX $\phi$, in kV during Dec 4, 2021 Eclipse", 
        "",
        num_subplots=3
    )
    rti.unique_times = ddates[0]
    ax = rti.addParamPlot(
        "mcm", drads[0].df, 7, "(a) 03 December, 2021", 
        p_max=500, p_min=100,
        cmap="GnBu", cbar=True,
        xlabel="",
    )
    ax.set_xlim([dt.datetime(2021,12,3,5), dt.datetime(2021,12,3,10)])
    ax.text(0.05, 1.05, "mcm/7", ha="left", va="center", transform=ax.transAxes)
    rti.unique_times = ddates[1]
    ax = rti.addParamPlot(
        "mcm", drads[1].df, 7, "(b) 04 December, 2021", 
        p_max=500, p_min=100,
        cmap="GnBu", cbar=False,
        xlabel="",
    )

    import eutils
    p = eutils.get_rti_eclipse(
        [dt.datetime(2021,12,4,5)+dt.timedelta(seconds=30*i) for i in range(5*60*2)],
        radar.get_lat_lon_along_beam(7)[0], radar.get_lat_lon_along_beam(7)[1]
    )
    cs = ax.contour(
        [dt.datetime(2021,12,4,5)+dt.timedelta(seconds=30*i) for i in range(5*60*2)],
        np.arange(76),
        p.T,
        colors="k", 
        linewidths=0.8,
        levels=[0.2, 0.4, 0.6, 0.75, 1.0],
        zorder=1, alpha=0.6,
    )
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    ax.set_xlim([dt.datetime(2021,12,4,5), dt.datetime(2021,12,4,10)])
    rti.unique_times = ddates[2]
    ax = rti.addParamPlot(
        "mcm", drads[2].df, 7, "(c) 05 December, 2021", 
        p_max=500, p_min=100,
        cmap="GnBu", cbar=False,
        xlabel="Time, UT",
    )
    ax.set_xlim([dt.datetime(2021,12,5,5), dt.datetime(2021,12,5,10)])
    
    rti.save("figures_2021_Special/rti_mcm_bgc.png")
    rti.close()
    return


def create_map_plot_cpcp_sd(
    extent=[-180, 180, -90, -50], 
    plt_lats = np.arange(-90, -49, 10), cb=False, mark_lon=-50,
    central_longitude=80, central_latitude=-70.0,
):
    # Ream MCM and DMSP
    from readdmsp import read_1D_dmsp_datasets
    dmspdata_south_boundary = read_1D_dmsp_datasets()
    dates = [dt.datetime(2021,12,4,6),dt.datetime(2021,12,4,10)]
    radar = Radar("mcm", dates, type="fitacf")
    radar.calculate_ground_range()
    df = radar.df.copy()
    df["unique_tfreq"] = df.tfreq
    radar.df = df
    # Read others
    from readmix import get_2D_data, get_sd_data, get_imfs, get_sd_dot_loc
    fan = Fan(
        [], dt.datetime(2021,12,4), f"", cb=cb,
        central_longitude=central_longitude, 
        central_latitude=central_latitude, extent=extent,
        plt_lats=plt_lats, nrows=3, ncols=4, sup_title=False,
        mark_lon=mark_lon, coord="geo"
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
        o = radar.df.copy()
        o = o[
            (o.time>=date)
            & (o.time<=date+dt.timedelta(minutes=1))
        ]
        utils.setsize(12)
        fan.date = date
        ax = fan.add_axes(add_coords=j==0, add_time=False)
        ax.overlay_eclipse(j==len(dates)-1)

        fan.generate_fov(
            "mcm", o, ax=ax, cbar=(j==7),
            # eclipse_cb=j==len(dates)-1, 
            eclipse_cb=False,
            p_max=500, p_min=300,
            xOffset=-1, yOffset=-3, 
            maxGate=75,
            label="Velocity, m/s", col="b",
            cmap="GnBu",
        )

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
        
        imfs = get_imfs(date)

        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="darkgreen", lw=0.5,)
        XYZ = fan.proj.transform_points(
            fan.geo, 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"], 
            dmspdata_south_boundary["MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE"]
        )
        ax.plot(XYZ[:, 0], XYZ[:, 1], ls="--", color="m", lw=0.5)
        ax.scatter(
            164.24,-74.62, 
            s=20,
            marker="^",
            color="k",
            zorder=2,
            transform=cartopy.crs.PlateCarree(),
            lw=0.8,
            alpha=0.8,
        )
        from read_digisonde import get_hv_by_date
        hv = get_hv_by_date(date)
        print("Digisonde HV:", hv)
        q = ax.quiver(
            np.array([[164.24]]), 
            np.array([[-74.62]]),
            np.array([[hv["VXF"]]]), np.array([[hv["VYF"]]]), 
            transform=cartopy.crs.PlateCarree(),
            headwidth=2, headlength=2, scale=1500, color="m", 
            zorder=3
        )
        if j==0:
            qk = ax.quiverkey(
                q,
                X=1.05,
                Y=0.8,
                U=500,
                angle=90,
                label="500 m/s",
                labelpos="E",
                coordinates="axes",
                labelsep=0.05
            )
            # Shrink and rotate the quiver key label for compact layout
            qk.text.set_fontsize("x-small")
            qk.text.set_rotation(90)

        data, lats, lons = get_sd_data(date)
        glat, glon = get_sd_dot_loc(date)

        xyz = fan.proj.transform_points(
            fan.geo, 
            glon, 
            glat
        )
        ax.scatter(
            xyz[:, 0], xyz[:, 1],
            s=0.3,
            marker="s",
            color="k",
            edgecolor="k",
            zorder=8,
            lw=0.8,
            alpha=0.3,
        )

        XYZ = fan.proj.transform_points(
            fan.geo, 
            lons, 
            lats
        )
        data = np.ma.masked_where(data==0., data)
        im = ax.contourf(
            XYZ[:, :, 0], XYZ[:, :, 1], data,
            cmap="RdBu",
            alpha=0.8,
            # vmax=45, vmin=-45
            levels=np.arange(-45, 46, 15),
        )
        txt = fr"$\phi_0$={np.round(np.max(data)-np.min(data),1)} kV" + "\n"
        txt = txt + fr"$\theta$={np.round(imfs['IMF.tilt, deg'].iloc[0],1)}$^\circ$"+ "\n"
        txt = txt + fr"$|B|$={np.round(imfs['IMF.B, nT'].iloc[0],1)} nT"+ "\n"
        txt = txt + fr"n={np.round(imfs['nvecs'].iloc[0],1)}"+ "\n"
        txt = txt + (r"$Vz_{jb}$=%.1f m/s"%(hv['VZF']))
        ax.text(0.05, 1.05, f"({chr(ord('A')+j)}) {date.strftime('%H:%M UT')}", ha="left", va="top", transform=ax.transAxes, fontdict={"size": "xx-small", "weight": "bold", "color": "k"})
        ax.text(0.05, 0.95, txt, ha="left", va="top", transform=ax.transAxes, fontdict={"size": 6, "color": "k"})
        if j==3:
            utils.setsize(10)
            cpos = [1.05, 0.1, 0.025, 0.6]
            cax = ax.inset_axes(cpos, transform=ax.transAxes)
            cb = fan.fig.colorbar(im, ax=ax, cax=cax)
            utils.setsize(10)
            cb.set_label(r"$\Phi$ [SuperDARN], kV")


    fan.fig.subplots_adjust(hspace=0.1, wspace=0.02)
    fan.save(f"figures_2021_Special/Maps_cpcp.png")
    fan.close()
    return


def create_DMSP_fluxplots():
    return

def create_rti_plots(
    rad_beams_ch_freq, dates, beam=15, yscale="srange", 
    range=[0, 4500]
):
    date = dates[1].strftime("%d %b, %Y") if dates[0].day == dates[1].day else dates[0].strftime("%d-") + dates[1].strftime("%d %b, %Y")
    rti = RangeTimePlot(
        range, 
        dates, 
        date, 
        len(rad_beams_ch_freq),
        font="sans-sarif",
    )
    for j, rad_beam in enumerate(rad_beams_ch_freq):
        rad, beam, channel, tfreq = rad_beam[0], rad_beam[1], rad_beam[2], rad_beam[3]
        tx = tfreq if channel == 2 else tfreq + 0.2 
        title = fr"({chr(97+j)}) Rad: {rad} / Beam: {beam} / ch: {channel} / $f_0$: {tfreq} MHz"
        radar = Radar(rad, dates, type="fitacf")
        radar.calculate_ground_range()
        df = radar.df.copy()
        logger.info(f"Reading radar: {rad} / Beam: {beam} / Unique: {df.tfreq.unique()}, {df.channel.unique()}, {df.bmnum.unique()}")
        if channel:
            df = df[df.channel==channel]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        v, tf = np.array(df.v), np.array(df.unique_tfreq)
        v[tf==10.5] *= -1
        df.v = v
        if tfreq: 
            df = df[df.unique_tfreq==tfreq]
        ax = rti.addParamPlot(
            rad, df, 
            beam, title=title,
            p_max=50, p_min=-100,
            xlabel="Time, UT" if j==len(rad_beams_ch_freq)-1 else "", 
            ylabel="Slant Range, km", 
            zparam="v", label=r"Velocity, $ms^{-1}$",
            cmap="RdBu", cbar=j==0, add_gflg=False,
            yparam="srange", kind="scatter"
        )
        # rti.add_conjugate_eclipse(rad, beam, dates, ax)
        # ddates = [
        #     dates[0] + dt.timedelta(minutes=int(1*i)) 
        #     for i in np.arange(int((dates[1]-dates[0]).total_seconds()/60)+1)
        # ]
        rti.overlay_eclipse_shadow(
            rad, beam, dates, ax, j==len(rad_beams_ch_freq)-1, 0.05
        )
        ax.set_ylim(range)
        ax.set_xlim(dates)
    rti.save(f"figures_2021_Special/rti.{rad}-{beam}.png")
    rti.close()
    return
