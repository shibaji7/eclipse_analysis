#!/usr/bin/env python3

"""plots.py: Plots dataset"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import Size, SubplotDivider
from mpl_toolkits.axes_grid1.mpl_axes import Axes
import datetime as dt
#plt.style.use(["science", "ieee"])
import mplstyle
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from matplotlib.projections import polar
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
from pylab import gca

def gerenate_fov_plot(rads, beam, date):
    import utils
    from fanUtils import Fan
    fan = Fan(rads[0], dt.datetime(2017,8, 21, 17), fig_title=f"{'%02d'%beam}", cs=False)
    ax = fan.add_axes()
    ax.overaly_eclipse_path(year=2017)
    ax.overaly_eclipse_path(year=2024)
    for rad in rads:
        fan.generate_fov(rad, [], ax=ax, beams=[beam])
    fan.save(filepath=utils.get_folder(rads[0], beam, date)+"/fov.png")
    return

def create_movie(folder, fname, file_ext=".png", movie_ext="mp4"):
    import os
    if os.path.exists(f"{folder}{fname}.{movie_ext}"):
        os.remove(f"{folder}{fname}.{movie_ext}")
    cmd = f"ffmpeg -framerate 1 -pattern_type glob -i '{folder}*{file_ext}' -c:v libx264 {folder}{fname}.{movie_ext}"
    os.system(cmd)
    return


def textHighlighted(
    xy,
    text,
    ax=None,
    color="k",
    fontsize=None,
    xytext=(0, 0),
    zorder=None,
    text_alignment=(0, 0),
    xycoords="data",
    textcoords="offset points",
    **kwargs
):
    """
    Plot highlighted annotation (with a white lining)

    Parameters
    ----------
    xy : position of point to annotate

    text : str text to show
    ax : Optional[ ]
    color : Optional[char]
    text color; deafult is "k"
    fontsize : Optional [ ] text font size; default is None
    xytext : Optional[ ] text position; default is (0, 0)
    zorder : text zorder; default is None
    text_alignment : Optional[ ]
    xycoords : Optional[ ] xy coordinate[1]; default is "data"
    textcoords : Optional[ ] text coordinate[2]; default is "offset points"
    **kwargs :

    Notes
    -----
    Belongs to class rbspFp.

    References
    ----------
    [1] see `matplotlib.pyplot.annotate
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.annotate>`)
    [2] see `matplotlib.pyplot.annotate
    <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.annotate>`)
    """

    if ax is None:
        ax = gca()
    text_path = mp.text.TextPath((0, 0), text, size=fontsize, **kwargs)
    p1 = matplotlib.patches.PathPatch(
        text_path,
        ec="w",
        lw=4,
        fc="w",
        alpha=0.7,
        zorder=zorder,
        transform=mp.transforms.IdentityTransform(),
    )
    p2 = matplotlib.patches.PathPatch(
        text_path,
        ec="none",
        fc=color,
        zorder=zorder,
        transform=mp.transforms.IdentityTransform(),
    )
    offsetbox2 = matplotlib.offsetbox.AuxTransformBox(mp.transforms.IdentityTransform())
    offsetbox2.add_artist(p1)
    offsetbox2.add_artist(p2)
    ab = mp.offsetbox.AnnotationBbox(
        offsetbox2,
        xy,
        xybox=xytext,
        xycoords=xycoords,
        boxcoords=textcoords,
        box_alignment=text_alignment,
        frameon=False,
    )
    ab.set_zorder(zorder)
    ax.add_artist(ab)
    return


def addColorbar(mappable, ax):
    """
    Append colorbar to axes

    Parameters
    ----------
    mappable : a mappable object
    ax : an axes object

    Returns
    -------
    cbax : colorbar axes object

    Notes
    -----
    This is mostly useful for axes created with :func:`curvedEarthAxes`.
    """
    fig1 = ax.get_figure()
    divider = SubplotDivider(fig1, *ax.get_geometry(), aspect=True)
    # axes for colorbar
    cbax = Axes(fig1, divider.get_position())
    h = [
        Size.AxesX(ax),  # main axes
        Size.Fixed(0.1),  # padding
        Size.Fixed(0.2),
    ]  # colorbar
    v = [Size.AxesY(ax)]
    _ = divider.set_horizontal(h)
    _ = divider.set_vertical(v)
    _ = ax.set_axes_locator(divider.new_locator(nx=0, ny=0))
    _ = cbax.set_axes_locator(divider.new_locator(nx=2, ny=0))
    _ = fig1.add_axes(cbax)
    _ = cbax.axis["left"].toggle(all=False)
    _ = cbax.axis["top"].toggle(all=False)
    _ = cbax.axis["bottom"].toggle(all=False)
    _ = cbax.axis["right"].toggle(ticklabels=True, label=True)
    _ = plt.colorbar(mappable, cax=cbax, shrink=0.7)
    return cbax


def curvedEarthAxes(
    rect=111,
    fig=None,
    minground=0.0,
    maxground=2000,
    minalt=0,
    maxalt=500,
    Re=6371.0,
    nyticks=5,
    nxticks=4,
):
    """
    Create curved axes in ground-range and altitude

    Parameters
    ----------
    rect : Optional[int] subplot spcification
    fig : Optional[pylab.figure object] (default to gcf)
    minground : Optional[float]
    maxground : Optional[int] maximum ground range [km]
    minalt : Optional[int] lowest altitude limit [km]
    maxalt : Optional[int] highest altitude limit [km]
    Re : Optional[float] Earth radius in kilometers
    nyticks : Optional[int] Number of y axis tick marks; default is 5
    nxticks : Optional[int] Number of x axis tick marks; deafult is 4

    Returns
    -------
    ax : matplotlib.axes object containing formatting
    aax : matplotlib.axes objec containing data
    """

    ang = maxground / Re
    minang = minground / Re
    angran = ang - minang
    angle_ticks = [(0, "{:.0f}".format(minground))]
    while angle_ticks[-1][0] < angran:
        tang = angle_ticks[-1][0] + 1.0 / nxticks * angran
        angle_ticks.append((tang, "{:.0f}".format((tang - minang) * Re)))
    grid_locator1 = FixedLocator([v for v, s in angle_ticks])
    tick_formatter1 = DictFormatter(dict(angle_ticks))

    altran = float(maxalt - minalt)
    alt_ticks = [(minalt + Re, "{:.0f}".format(minalt))]
    while alt_ticks[-1][0] < Re + maxalt:
        alt_ticks.append(
            (
                altran / float(nyticks) + alt_ticks[-1][0],
                "{:.0f}".format(altran / float(nyticks) + alt_ticks[-1][0] - Re),
            )
        )
    _ = alt_ticks.pop()
    grid_locator2 = FixedLocator([v for v, s in alt_ticks])
    tick_formatter2 = DictFormatter(dict(alt_ticks))
    tr_rotate = Affine2D().rotate(np.pi / 2 - ang / 2)
    tr_shift = Affine2D().translate(0, Re)
    tr = polar.PolarTransform() + tr_rotate

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr,
        extremes=(0, angran, Re + minalt, Re + maxalt),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=tick_formatter1,
        tick_formatter2=tick_formatter2,
    )
    if not fig:
        fig = plt.figure(figsize=(8, 3))
    ax1 = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    # adjust axis
    ax1.axis["left"].label.set_text(r"Height [km]")
    ax1.axis["bottom"].label.set_text(r"Ground range [km]")
    ax1.invert_xaxis()
    ax1.minground = minground
    ax1.maxground = maxground
    ax1.minalt = minalt
    ax1.maxalt = maxalt
    ax1.Re = Re
    fig.add_subplot(ax1, transform=tr)
    # create a parasite axes whose transData in RA, cz
    aux_ax = ax1.get_aux_axes(tr)
    # for aux_ax to have a clip path as in ax
    aux_ax.patch = ax1.patch
    # but this has a side effect that the patch is drawn twice, and possibly
    # over some other artists. So, we decrease the zorder a bit to prevent this.
    ax1.patch.zorder = 0.9
    return ax1, aux_ax


def get_polar(d, Re=6371.0):
    """Convert to polar coordinates"""
    th, r = to_polar(d.ground_range, d.height, Re)
    return th, r


def to_polar(grange, height, Re=6371.0):
    th = np.array(grange) / Re
    r = np.array(height) + Re
    return th, r


def plot_rays(
    dic,
    fig_fname,
    rto,
    txt,
    maxground=1500,
    maxalt=500,
    step=1,
    showrefract=False,
    nr_cmap="jet_r",
    nr_lim=[0.0, 0.1],
    raycolor="0.3",
    title=True,
    zorder=2,
    alpha=0.4,
    fig=None,
    rect=111,
    ax=None,
    aax=None,
):
    """
    Plot ray paths

    Parameters
    ----------
    dic: str location of the data files
    time: datetime.datetime time of rays
    ti: int time index
    beam: beam number
    maxground : Optional[int] maximum ground range [km]
    maxalt : Optional[int] highest altitude limit [km]
    step : Optional[int] step between each plotted ray (in number of ray steps)
    showrefract : Optional[bool] show refractive index along ray paths (supersedes raycolor)
    nr_cmap : Optional[str] color map name for refractive index coloring
    nr_lim : Optional[list, float] refractive index plotting limits
    raycolor : Optional[float] color of ray paths
    title : Optional[bool] Show default title
    zorder : Optional[int]
    alpha : Optional[int]
    fig : Optional[pylab.figure] object (default to gcf)
    rect : Optional[int] subplot spcification
    ax : Optional[ ] Existing main axes
    aax : Optional[ ] Existing auxialary axes

    Returns
    -------
    ax : matplotlib.axes object containing formatting
    aax : matplotlib.axes object containing data
    cbax : matplotlib.axes object containing colorbar
    """
    if not ax and not aax:
        ax, aax = curvedEarthAxes(
            fig=fig, rect=rect, maxground=maxground, maxalt=maxalt
        )
    # Plot density
    th, r = to_polar(
        rto.bearing_object["dist"],
        rto.bearing_object["heights"],
    )
    pf = np.sqrt(80.6164e-6 * rto.density)
    im = aax.pcolormesh(
        th,
        r,
        pf,
        cmap="plasma",
        alpha=0.8,
        vmax=6,
        vmin=0,
    )
    pos = aax.get_position()
    cpos = [
        pos.x1 + 0.025,
        pos.y0 + 0.0125,
        0.015,
        pos.height * 0.5,
    ]  # this list defines (left, bottom, width, height)
    cs = aax.contour(
        th,
        r,
        rto.p.T,
        levels=np.linspace(0,1,21),
        color="k",
        alpha=0.5,
    )
    fig = ax.get_figure()
    aax.clabel(cs, inline=1, fontsize=10, fmt="%.2f", colors="k")
    cax = fig.add_axes(cpos)
    cbax = fig.colorbar(
        im, cax, spacing="uniform", orientation="vertical", cmap="plasma"
    )
    _ = cbax.set_label(r"$f_0$ [MHz]")
    unique_lables = []
    # Plot rays
    elvs = rto.rays.elvs
    for i, elv in enumerate(elvs[::2]):
        ray_path_data, ray_data = (
            rto.rays.ray_path_data[elv],
            rto.rays.simulation[elv]["ray_data"]
        )
        th, r = get_polar(ray_path_data)
        if not showrefract:
            ray_label = ray_data["ray_label"]
            aax.plot(th, r, c=raycolor, zorder=zorder, alpha=alpha, ls="-", lw=0.5)
            col = "k" if ray_label == 1 else "r"
            unique_lables.append(ray_label)
            if ray_label in [-1, 1]:
                aax.scatter([th[-1]], [r[-1]], marker="s", s=3, color=col)
        else:
            points = np.array([th, r]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lcol = LineCollection(segments, zorder=zorder, alpha=alpha)
            _ = lcol.set_cmap(nr_cmap)
            _ = lcol.set_norm(plt.Normalize(*nr_lim))
            _ = lcol.set_array(v)
            _ = aax.add_collection(lcol)
    if title:
        event = rto.event
        stitle = "%s UT" % event.strftime("%Y-%m-%d %H:%M")
        aax.text(0.95, 1.01, stitle, ha="right", va="center", transform=ax.transAxes)
        aax.text(0.05, 1.01, txt, ha="left", va="center", transform=ax.transAxes)
    if showrefract:
        cbax = addColorbar(lcol, ax)
        _ = cbax.set_ylabel(r"$\eta$")
    else:
        cbax = None

    ax.beam = rto.beam
    fig = ax.get_figure()
    fig.savefig(
        fig_fname,
        bbox_inches="tight",
    )
    plt.close()
    print(set(unique_lables))
    return
