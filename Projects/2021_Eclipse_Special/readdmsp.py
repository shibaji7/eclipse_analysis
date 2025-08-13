from xarray import open_dataset

def read_dmsp(file_path="database/DMSP2021/dmspf18_ssusi_edr-aurora_2021333T084722-2021333T102913-REV62485_vA8.2.0r000.nc"):
    ds = open_dataset(file_path)
    return ds

def read_1D_dmsp_datasets(file_path, variables=["SOUTH_GEOGRAPHIC_LATITUDE", "SOUTH_GEOGRAPHIC_LONGITUDE"]):
    ds = read_dmsp(file_path)
    data = {var: ds[var].values for var in variables}
    ds.close()
    return data

if __name__ == "__main__":
    read_dmsp()