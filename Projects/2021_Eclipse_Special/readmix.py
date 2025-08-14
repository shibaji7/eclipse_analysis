from xarray import open_dataset
import datetime as dt

def read_mix(file_path=None):
    if file_path is None:
        # Default file path for DMSP data
        # Adjust this path as necessary for your environment
        file_path = "database/mix_gitm_south_20211204.nc"
    ds = open_dataset(file_path)
    print(ds.Pot.values[0, :, :].max()-ds.Pot.values[0, :, :].min())
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


if __name__ == "__main__":
    get_pot_drop()