import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]

import mplstyle

import matplotlib as mpl

import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np
from loguru import logger

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import eutils as utils


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, nrang, unique_times, fig_title, num_subplots=3, font="sans-serif"):
        plt.rcParams["font.family"] = font
        if font == "sans-serif":
            plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]
        dpi = 180 #if num_subplots==2 else 180 - ( 40*(num_subplots-2) )
        self.nrang = nrang
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=dpi) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=15)
        mpl.rcParams.update({"xtick.labelsize": 12, "ytick.labelsize":12, "font.size":12})
        utils.setsize(12)
        return
    
    def addParamPlot(self, rad, df, beam, title, p_max=100, p_min=-100, xlabel="Time UT",
             ylabel="Range gate", zparam="v", label="Velocity [m/s]", cmap="jet_r", 
             cbar=False, omni=None, add_gflg=False):
        ax = self._add_axis()
        logger.info(f"Unique beams: {df.bmnum.unique()}")
        df = df[df.bmnum==beam]
        if beam not in df.bmnum.unique():
            logger.error(f"Beam {beam} was not sounded!")
        X, Y, Z = utils.get_gridded_parameters(df, xparam="time", yparam="srange", zparam=zparam)
        if add_gflg:
            Xg, Yg, Zg = utils.get_gridded_parameters(df, xparam="time", yparam="srange", zparam="gflg")
        cmap = cmap
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim(self.nrang)
        ax.set_ylabel(ylabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        if add_gflg:
            Zx = np.ma.masked_where(Zg==0, Zg)
            ax.pcolormesh(Xg, Yg, Zx.T, lw=0.01, edgecolors="None", cmap="gray",
                        vmax=2, vmin=0, shading="nearest")
            Z = np.ma.masked_where(Zg==1, Z)
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest", zorder=3)
        else:
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest", zorder=3)
        ax.tick_params(direction="out", which="both")
        self.overlay_eclipse_shadow(rad, beam, self.unique_times, ax)
        if omni is None: 
            if cbar: self._add_colorbar(im, ax, cmap, label=label)
            return ax
        else: 
            if cbar: self._add_colorbar(im, ax, cmap, label=label, dx=0.15)
            t_ax = self.overlay_omni(ax, omni)
            return ax, t_ax
            
    def overlay_eclipse_shadow(self, rad, beam, dates, ax):
        ddates = [
            dates[0]+dt.timedelta(minutes=i) 
            for i in range(int((dates[1]-dates[0]).total_seconds()/60))
        ]
        import pydarn
        hdw = pydarn.read_hdw_file(rad)
        fov = pydarn.Coords.GEOGRAPHIC(hdw.stid)
        glat, glon = fov[0][:101, beam], fov[1][:101, beam]
        p = utils.get_rti_eclipse(ddates, glat, glon)
        srange = 180 + (45 * np.arange(101))
        obs = np.copy(p)
        obs[obs>1.] = np.nan
        ax.contourf(
            ddates,
            srange,
            obs.T,
            cmap="Blues", alpha=0.6,
            levels=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
        )
        obs = np.copy(p)
        obs[obs<=1.] = np.nan
        ax.pcolormesh(
            ddates, srange, obs.T, lw=0.01, edgecolors="None", cmap="gray_r",
            vmax=1, vmin=0, shading="nearest", alpha=0.4, zorder=1
        )
        return
    
    def overlay_omni(self, ax, omni):
        t_ax = ax.twinx()
        t_ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        t_ax.set_ylim(0, 360)
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        t_ax.xaxis.set_major_locator(hours)
        t_ax.set_ylabel(r"IMF Clock Angle ($\theta_c$)"+"\n[in degrees]", 
            fontdict={"size":12, "fontweight": "bold"})
        t_ax.plot(
            omni.date, omni.Tc, 
            ls="-", lw=1.6, color="m"
        )
        t_ax.plot(
            omni.date, omni.Tc, 
            ls="--", lw=0.6, color="k"
        )
        return t_ax

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=12)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    def _add_colorbar(self, im, ax, colormap, label="", dx=0.01):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * dx, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(im, cax,
                   spacing="uniform",
                   orientation="vertical", 
                   cmap=colormap)
        cb2.set_label(label)
        return