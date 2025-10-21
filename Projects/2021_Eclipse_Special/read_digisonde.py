from xarray import open_dataset
import datetime as dt
import numpy as np
import pandas as pd

DataBase_Path = "/home/chakras4/OneDrive/Chakras4/Projects/Chakraborty.Projects/byProjects/2024 Eclipse Project/Datasets/2021_Dec_Eclipse/Jang_Bogo/"

def read_digisonde(file_path=None):
    if file_path is None:
        # Default file path for Digisonde data
        # Adjust this path as necessary for your environment
        file_path = DataBase_Path + "result_2021_12_04.nc"
    ds = open_dataset(file_path)
    return ds

def read_velocity_data(file_path=None, component="VXF"):
    ds = read_digisonde(file_path)
    velocity_data = ds[component].values
    velocity_err = ds[f"d{component}"].values
    # Read date and time information
    years = ds["yr"].values
    dDays = ds["dDay"].values
    dates = [
        dt.datetime(yr, 1, 1) + dt.timedelta(days=dy - 1)
        for yr, dy in zip(years, dDays)
    ]
    ds.close()
    return dates, velocity_data, velocity_err

def create_base_line(
    files = [
        "result_2021_11_30.nc",
        "result_2021_12_02.nc",
        "result_2021_12_03.nc",
        "result_2021_12_05.nc",
        "result_2021_12_06.nc",
        "result_2021_12_07.nc",
        "result_2021_12_08.nc",
        "result_2021_12_09.nc",
    ],
    folder=DataBase_Path,
    component="VXF",
):
    Component_Vel, Component_Vel_Err = [], []
    for f in files:
        file_path = folder + f
        dates, velocity_data, velocity_err = read_velocity_data(file_path, component)
        # Process or store the data as needed
        Component_Vel.append(velocity_data)
        Component_Vel_Err.append(velocity_err)
    Component_Vel = np.nanmean(np.array(Component_Vel), axis=0)
    Component_Vel_Err = np.nanmean(np.array(Component_Vel_Err), axis=0)
    print("Base line data shape:", Component_Vel.shape)
    return Component_Vel, Component_Vel_Err

def get_locational_info(file_path="-74.62lat_-164.24lon_20211204070000_20211204091500.nc"):
    file_path = DataBase_Path + file_path
    ds = open_dataset(file_path)
    lat = ds["glat"].values.item()
    lon = ds["glon"].values.item()
    time, eof = (
        [
            pd.Timestamp(t).to_pydatetime() 
            for t in ds["time"].values
        ], 
        ds["211"].values
    )
    ds.close()
    return lat, lon, time, eof

def consolidate_data(component="VXF"):
    dates, velocity_data, velocity_err = read_velocity_data(component=component)
    Component_Vel, Component_Vel_Err = create_base_line(component=component)
    lat, lon, time, eof = get_locational_info()
    return (
        dates, 
        velocity_data, 
        velocity_err, 
        Component_Vel, 
        Component_Vel_Err,
        lat, lon, time, eof
    )

def get_hv_by_date(dx=dt.datetime(2021,12,4,7,30)):
    dates, vxf, _ = read_velocity_data(component="VXF")
    _, vyf, _ = read_velocity_data(component="VYF")
    _, vzf, _ = read_velocity_data(component="VZF")
    idx = np.argmin(np.abs(np.array(dates) - dx))
    print("Closest date found:", dates[idx])
    data = {
        "VXF": np.round(vxf[idx],1), 
        "VYF": np.round(vyf[idx],1),
        "VZF": np.round(vzf[idx],1),
    }
    return data

get_hv_by_date()