from xarray import open_dataset
import datetime as dt

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

def get_pot_drop(file_path=None, var="Pot"):
    ds = read_mix(file_path)
    mjd_epoch = dt.datetime(1858, 11, 17, 0, 0, 0)
    time = [mjd_epoch+dt.timedelta(d) for d in ds.time.values]
    varsl = []
    for i in range(len(time)):
        varsl.append(ds[var].values[i, :, :].max()-ds[var].values[i, :, :].min())
    ds.close()
    return varsl, time

def get_2D_data(date, file_path=None,var="J_par"):
    ds = read_mix(file_path)
    times = mdj(ds.time.values)
    i = times.index(date)
    data, lats, lons = (
        ds[var].values[i, :, :], 
        ds["geo_lat"].values[i, :, :], 
        ds["geo_lon"].values[i, :, :]
    )

    ds.close()
    return (data, lats, lons)


def read_sd(file=None):
    if file is None:
        file = "/home/chakras4/OneDrive/Chakras4/SuperDARN-Data-Share/Chakraborty2021/20211204.south.nc"
    print(file)
    ds = open_dataset(file)
    return ds

def get_sd_data(date, file=None, var="fparam.pot_arr"):
    ds = read_sd(file)
    times = get_sd_time(ds["map.stime"].values)
    i = times.index(date)
    print(f"Fetching data for {date}, index: {i}")
    data = ds[var].values[i, :, :]
    lats, lons = ds["fparam.lat_pot"].values, ds["fparam.lon_pot"].values
    print(ds)
    ds.close()
    return (data, lats, lons)


if __name__ == "__main__":
    get_sd_data(dt.datetime(2021,12,4,7,40))