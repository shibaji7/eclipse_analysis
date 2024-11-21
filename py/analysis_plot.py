"""
    This python module is used to analyze and plot the dataset.
"""
from math import radians
import os
import datetime as dt
import pandas as pd
import pydarn
import numpy as np


from read_fitacf import Radar
from plot import RangeTimePlot
from fan import Fan


def generate_location_file(rad="pgr", beams=[]):
    os.makedirs("figures/", exist_ok=True)
    hdw = pydarn.read_hdw_file(rad)
    lat, lon = pydarn.Coords.GEOGRAPHIC(hdw.stid)
    beams, gates = hdw.beams, hdw.gates
    rec = []
    for gate in range(lat.shape[0]):
        for beam in range(lat.shape[1]):
            rec.append({
                "beam": beam,
                "gate": gate,
                "gdlat": lat[gate, beam],
                "glong": lon[gate, beam]
            })
    rec = pd.DataFrame.from_records(rec)
    rec.to_csv(f"figures/{rad}.csv", header=True, index=False)
    return

def gsMapSlantRange(slant_range, altitude=None, elevation=None):
    """Calculate the ground scatter mapped slant range.
    See Bristow et al. [1994] for more details. (Needs full reference)
    Parameters
    ----------
    slant_range
        normal slant range [km]
    altitude : Optional[float]
        altitude [km] (defaults to 300 km)
    elevation : Optional[float]
        elevation angle [degree]
    Returns
    -------
    gsSlantRange
        ground scatter mapped slant range [km] (typically slightly less than
        0.5 * slant_range.  Will return -1 if
        (slant_range**2 / 4. - altitude**2) >= 0. This occurs when the scatter
        is too close and this model breaks down.
    """
    Re = 6731.1

    # Make sure you have altitude, because these 2 projection models rely on it
    if not elevation and not altitude:
        # Set default altitude to 300 km
        altitude = 300.0
    elif elevation and not altitude:
        # If you have elevation but not altitude, then you calculate altitude,
        # and elevation will be adjusted anyway
        altitude = np.sqrt(Re ** 2 + slant_range ** 2 + 2. * slant_range * Re *
                           np.sin(np.radians(elevation))) - Re
    if (slant_range**2) / 4. - altitude ** 2 >= 0:
        gsSlantRange = Re * \
            np.arcsin(np.sqrt(slant_range ** 2 / 4. - altitude ** 2) / Re)
        # From Bristow et al. [1994]
    else:
        gsSlantRange = -1
    return gsSlantRange

class Analysis(object):

    def __init__(
        self,
        rad, 
        beam, 
        dates, 
        type="fitacf3",
        font="sans-sarif",
        rti_panels = 2
    ):
        self.rad = rad
        self.dates = dates
        self.beam = beam
        self.type = type
        self.font = font
        os.makedirs("figures/", exist_ok=True)
        self.radar = Radar(rad, dates=dates, type=type)
        self.rti_panels = rti_panels
        self.iniRTI()
        return
    
    def get_unique_freq(self):
        df = self.radar.df.copy()
        df = df[
            (df.time>=self.dates[0]) & 
            (df.time<=self.dates[-1])
        ]
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        tf = df.unique_tfreq.unique()
        if (10. in tf) and (10.5 in tf):
            tf = np.delete(tf, np.where(tf == 10.)[0])
        tf = ",".join(str(x) for x in tf)
        del df
        return tf

    def iniRTI(
        self,
        range = [0,4000],
        title = None
    ):
        if title is None:
            tf = self.get_unique_freq()
            d0, d1 = self.dates[0], self.dates[1]-dt.timedelta(seconds=60)
            date = self.dates[0].strftime("%d %b, %Y") if d0.day==d1.day else \
                self.dates[0].strftime("%d-") + self.dates[1].strftime("%d %b, %Y")
            title = fr"Rad: {self.rad} / Beam: {self.beam} / Date:  {date}"
        # from gps import GPS
        # self.gps = GPS()
        # lats, lons = self.radar.get_lat_lon_along_beam(self.beam)
        # self.tecv = self.gps.get_tec_along_beam(lats, lons)
        self.rti = RangeTimePlot(
            range, 
            self.dates, 
            title, 
            self.rti_panels,
            font=self.font
        )
        return

    def generateRTI(
        self, 
        params = [
            {
                "col": "jet",
                "ylim": [800, 2000],
                "xlim": None,
                "title": "", 
                "p_max":30, "p_min":3, 
                "xlabel": "", "ylabel": "Slant Range [km]",
                "zparam":"p_l", "label": "Power [dB]",
                "cmap": "jet", "cbar": True,
            }
        ],
        tfreq=None, gflg=None, channel=None        
    ):
        for i, param in enumerate(params):
            if "_gs" in param["zparam"]:
                param["zparam"] = param["zparam"].replace("_gs", "")
                self.df["slist"] = np.array(self.df["gsMap"])
            ax = self.rti.addParamPlot(
                self.rad,
                self.df, 
                self.beam, param["title"], 
                p_max=param["p_max"], p_min=param["p_min"],
                xlabel=param["xlabel"], ylabel=param["ylabel"], 
                zparam=param["zparam"], label=param["label"],
                cmap=param["cmap"], cbar=param["cbar"], add_gflg=param["gflg"]
            )
            ax.set_ylim(param["ylim"])
            if param["xlim"]: ax.set_xlim(param["xlim"])
            if i == 0:
                ax.text(0.99, 1.05, self.filter_summ, va="bottom",
                        ha="right", transform=ax.transAxes)
        # im = ax.pcolormesh(
        #     self.tecv["X"], self.tecv["Y"], self.tecv["Z"].T, 
        #     lw=0.01, edgecolors="None", cmap="YlOrRd",
        #     vmax=40, vmin=10, shading="nearest", alpha=0.4
        # )
        # self.rti._add_colorbar(im, ax, "YlOrRd", label="TEC", dx=0.15)
        self.rti.save(f"figures/rti.{self.rad}-{self.beam}.png")
        self.rti.close()
        return

    def generateFan(self, date, param="v", gflg=False):
        p_max, p_min = (30, -30) if param == "v" else (33, 0)
        cmap = "Spectral" if param == "v" else "plasma"
        label = "Velocity [m/s]" if param == "v" else "Power [dB]"
        fig_title = rf"$f_0=${self.get_unique_freq()} MHz" 
        frame = self.df[
            (self.df.time >= date)
            & (self.df.time < date + dt.timedelta(minutes=1))
        ]
        self.fan = Fan(
            [self.rad], date,
            fig_title
        )
        self.fan.generate_fov(
            self.rad, frame, p_name=param,
            p_max=p_max, p_min=p_min, cmap=cmap,
            label=label
        )
        self.fan.save(f"figures/fan.{self.rad}-{date.strftime('%H%M')}.png")
        self.fan.close()      
        return

    def filter_dataframe(
        self,
        gflg=None,
        tfreq=None,
        channel=None,
        change_vel_sign=False,
    ):
        self.filter_summ = ""
        self.df = self.radar.df.copy()
        if channel:
            self.df = self.df[self.df.channel==channel]
            channel = "a" if channel==1 else "b"
            self.filter_summ += f"Channel: {channel}\n"
        self.df["srange"] = (self.df.slist*self.df.rsep) + self.df.frang
        self.df = self.df[
            (self.df.time>=self.dates[0]) & 
            (self.df.time<=self.dates[-1])
        ]
        self.df["unique_tfreq"] = self.df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        print(f"Unique tfreq: {self.df.unique_tfreq.unique()}")
        if tfreq: 
            self.df = self.df[self.df.unique_tfreq==tfreq]
            self.filter_summ += r"$f_0$=%.1f MHz"%tfreq + "\n"
        if gflg: 
            self.df = self.df[self.df.gflg==gflg]
            self.filter_summ += r"IS/GS$\sim$%d"%gflg
        
        self.df["gsMap"] = self.df.apply(
            lambda row: gsMapSlantRange(row["slist"]), 
            axis = 1
        )
        if change_vel_sign:
            self.df["v"] = -1*self.df["v"]
        return
    
def genererate_RTI(
        rad, beam, dates, type, srange=[0, 4500],
        param_list=["v", "p_l"], gflg=False,
        tfreq=None, channel=None
):
    anl = Analysis(
        rad=rad,
        beam=beam, 
        dates=dates,
        type=type,
        font="sans-serif",
        rti_panels = len(param_list)
    )
    anl.filter_dataframe(channel=channel, tfreq=tfreq, change_vel_sign=(tfreq==10.5))
    params = []
    if "v" in param_list:
        params.append(
            {
                "ylim": srange,
                "title": "", 
                "xlim": dates,
                "p_max": 30, "p_min":-30, 
                "xlabel": "", "ylabel": "Slant Range [km]",
                "zparam":"v", "label": "Velocity [m/s]",
                "cmap": "jet_r", "cbar": True, "gflg":gflg
            }
        )
    if "p_l" in param_list:
        params.append(
            {
                "ylim": srange,
                "xlim": dates,
                "title": "", 
                "p_max": 30, "p_min":3, 
                "xlabel": "Time [UT]", "ylabel": "Slant Range [km]",
                "zparam":"p_l", "label": "Power [dB]",
                "cmap": "jet", "cbar": True, "gflg":False
            }
        )
    anl.generateRTI(params = params)
    return

def generate_fovt(rad, dates):
    fan = Fan(
        [rad], 
        dates[0].replace(hour=0,minute=0) + dt.timedelta(hours=7) + dt.timedelta(minutes=35),
        f""
    )
    fan.overlay_fovs(
        rad,
        #beams=[0,3,7,11,15] 
        beams=[15, 11, 7, 3, 0]
    )
    fan.save(f"figures/fanbeam.{rad}.png")
    fan.close() 
    return

def genererate_Fan(
    rad, dates, date, type="fitacf",
    param="v", tfreq=None, gflg=None, 
    channel=None
):
    anl = Analysis(
        rad=rad,
        beam=None, 
        dates=dates,
        type=type,
        font="sans-serif",
        rti_panels = len([param])
    )
    anl.filter_dataframe(gflg, tfreq, channel)
    anl.generateFan(date, param=param, gflg=False)
    return