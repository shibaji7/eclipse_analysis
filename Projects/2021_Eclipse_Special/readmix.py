from xarray import open_dataset
import datetime as dt
import numpy as np

def mdj(time):
    mjd_epoch = dt.datetime(1858, 11, 17, 0, 0, 0)
    time = [mjd_epoch+dt.timedelta(d) for d in time]
    return time

def get_sd_time(times):
    import pandas as pd
    times = [pd.to_datetime((t)) for t in times]
    return times

def read_mix(file_path=None):
    if file_path is None:
        # Default file path for DMSP data
        # Adjust this path as necessary for your environment
        file_path = "database/mix_gitm_south_20211204.nc"
    ds = open_dataset(file_path)
    return ds

def get_pot_drop(file_path=None, var="J_par"):
    ds = read_mix(file_path)
    mjd_epoch = dt.datetime(1858, 11, 17, 0, 0, 0)
    time = [mjd_epoch+dt.timedelta(d) for d in ds.time.values]
    varsl = []
    for i in range(len(time)):
        varsl.append(ds[var].values[i, :, :].max()-ds[var].values[i, :, :].min())
    ds.close()
    return varsl, time


def get_2D_data(date, file_path=None,var="J_par", intp=False):
    ds = read_mix(file_path)
    times = mdj(ds.time.values)
    i = times.index(date)
    data, lats, lons = (
        ds[var].values[i, :, :], 
        ds["geo_lat"].values[i, :, :], 
        ds["geo_lon"].values[i, :, :]
    )
    ds.close()
    if intp:
        lats, lons, data = spherical_interp_latlon(lats, lons, data)
    return (data, lats, lons)


import numpy as np
from scipy.interpolate import RectSphereBivariateSpline

def spherical_interp_latlon(jlats, jlons, Jpar, up=4):
    # jlats, jlons, Jpar shape: (nlat, nlon)

    lat = jlats[:, 0].astype(float)            # (nlat,)
    lon = jlons[0, :].astype(float)            # (nlon,)

    # Put lon in [0, 360), then sort both axes
    lon = (lon + 360.0) % 360.0
    lat_idx = np.argsort(lat)
    lon_idx = np.argsort(lon)

    lat = lat[lat_idx]
    lon = lon[lon_idx]
    z = Jpar[lat_idx, :][:, lon_idx]

    # Remove duplicate axis values (required by spline)
    lat, lat_keep = np.unique(lat, return_index=True)
    lon, lon_keep = np.unique(lon, return_index=True)
    z = z[lat_keep, :][:, lon_keep]

    # RectSphere uses colatitude u in (0, pi), longitude v in [0, 2pi)
    u = np.deg2rad(90.0 - lat)                 # colat
    v = np.deg2rad(lon)

    # Avoid exact poles
    eps = 1e-8
    u = np.clip(u, eps, np.pi - eps)

    # Spline requires strictly increasing u and v.
    u_order = np.argsort(u)
    u = u[u_order]
    z = z[u_order, :]
    u, u_keep = np.unique(u, return_index=True)
    z = z[u_keep, :]

    v_order = np.argsort(v)
    v = v[v_order]
    z = z[:, v_order]
    v, v_keep = np.unique(v, return_index=True)
    z = z[:, v_keep]

    if len(u) < 2 or len(v) < 2:
        raise ValueError("Not enough unique spherical grid points for interpolation.")

    # High-res target grid
    lat_hi = np.linspace(lat.min(), lat.max(), len(lat) * up)
    lon_hi = np.linspace(lon.min(), lon.max(), len(lon) * up)
    u_hi = np.deg2rad(90.0 - lat_hi)
    v_hi = np.deg2rad(lon_hi)
    u_hi = np.clip(u_hi, eps, np.pi - eps)
    if u_hi[0] > u_hi[-1]:
        u_hi = u_hi[::-1]
        lat_hi = lat_hi[::-1]

    # Spherical interpolation
    spl = RectSphereBivariateSpline(u, v, z, s=0.0)
    Jpar_hi = spl(u_hi, v_hi, grid=True)       # shape (len(lat_hi), len(lon_hi))

    LAT_hi, LON_hi = np.meshgrid(lat_hi, lon_hi, indexing="ij")
    print(LON_hi, LAT_hi)
    return LAT_hi, LON_hi, Jpar_hi

def read_sd(file=None):
    if file is None:
        file = "/home/chakras4/OneDrive/Chakras4/SuperDARN-Data-Share/Chakraborty2021/20211204.south.nc"
    ds = open_dataset(file)
    return ds

def get_map_level_datasets():
    import pydarn
    import bz2
    ds = read_sd()
    times = get_sd_time(ds["map.stime"].values)
    pot_drop, = ds["map.pot.drop"]
    f = "/home/chakras4/OneDrive/Chakras4/Projects/Chakraborty.Projects/byProjects/2024 Eclipse Project/2021 Special Eclipse/sd_map2/20211204.south.map2.bz2"
    with bz2.open(f) as fp: d = fp.read()
    reader = pydarn.SuperDARNRead(d, True)
    recs = reader.read_map()
    print(recs[0].keys())
    import pandas as pd
    o = pd.DataFrame()
    o["time, UT"], o["pot.drop, kV"] = times, pot_drop/1e3
    o["IMF.tilt, deg"], o["IMF.B, nT"], o["nrads"], o["nvecs"] = (
        [r["IMF.tilt"] for r in recs],
        [np.sqrt(r["IMF.Bx"]**2+r["IMF.By"]**2+r["IMF.Bz"]**2) for r in recs],
        [len(r["nvec"]) for r in recs],
        [len(r["vector.mlat"]) if "vector.mlat" in r else 0 for r in recs],
    ) 
    o["model"] = "TS18"
    print(recs[8]["vector.mlat"])
    o.to_csv("database/2021_pot_drop.csv", header=True, index=False, float_format="%g")
    ds.close()
    return

def get_imfs(date):
    import pandas as pd
    o = pd.read_csv("database/2021_pot_drop.csv", parse_dates=["time, UT"])
    o = o[o["time, UT"] == date]
    return o

def parse_pot_data():
    ds = read_sd()
    times = get_sd_time(ds["map.stime"].values)
    pot_drop = ds["map.pot.drop"]
    import pandas as pd
    o = pd.DataFrame()
    o["time, UT"], o["pot_drop, kV"] = times, pot_drop/1e3
    o.to_csv("database/2021_pot_drop.csv", header=True, index=False, float_format="%g")
    ds.close()
    return o

def get_sd_dot_loc(date, file=None):
    ds = read_sd(file)
    times = get_sd_time(ds["map.stime"].values)
    i = times.index(date)
    mlats = ds["fparam.mlats"].values[i, :]
    mlons = ds["fparam.mlons"].values[i, :]
    ds.close()
    import aacgmv2
    glats, glons = np.zeros_like(mlats), np.zeros_like(mlons)
    i = 0
    for xlat, xlon in zip(mlats, mlons):
        glats[i], glons[i], _ = aacgmv2.convert_latlon_arr(
            [xlat], [xlon], 300, date, 
            method_code="A2G"
        )
        i+=1
    return (glats, glons)

def get_sd_data(date, file=None, var="fparam.pot_arr"):
    ds = read_sd(file)
    times = get_sd_time(ds["map.stime"].values)
    i = times.index(date)
    print(f"Fetching data for {date}, index: {i}")
    data = ds[var].values[i, :, :]
    lats, lons = ds["fparam.lat_pot"].values, ds["fparam.lon_pot"].values
    print(lats.shape, lons.shape)
    hs = np.ones_like(lats)*300
    ds.close()
    import aacgmv2
    glats, glons = np.zeros_like(lats), np.zeros_like(lats)
    for i in range(lats.shape[0]):
        glats[i, :], glons[i, :], _ = aacgmv2.convert_latlon_arr(
            lats[i, :], lons[i,:], 300, date, 
            method_code="A2G"
        )
    return (data, glats, glons)


if __name__ == "__main__":
    # get_sd_data(dt.datetime(2021,12,4,7,40))
    get_map_level_datasets()
    # parse_pot_data()
