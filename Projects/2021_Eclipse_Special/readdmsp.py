from xarray import open_dataset

def read_dmsp(file_path=None):
    if file_path is None:
        # Default file path for DMSP data
        # Adjust this path as necessary for your environment
        file_path = "database/DMSP2021/dmspf17_ssusi_edr-aurora_2021338T070055-2021338T084246-REV77832_vA8.2.0r000.nc"
    ds = open_dataset(file_path)
    print(ds.DOY.values)
    return ds

def read_1D_dmsp_datasets(
    file_path=None, 
    variables=[
        "MODEL_SOUTH_GEOGRAPHIC_LATITUDE", 
        "MODEL_SOUTH_GEOGRAPHIC_LONGITUDE",
        "MODEL_SOUTH_POLAR_GEOGRAPHIC_LATITUDE",
        "MODEL_SOUTH_POLAR_GEOGRAPHIC_LONGITUDE",
    ]
):
    ds = read_dmsp(file_path)
    data = {var: ds[var].values for var in variables}
    ds.close()
    return data

def read_2D_dmsp_datasets(
    file_path=None, 
    variables=["SOUTH_GEOMAGNETIC_LATITUDE", "N_GEOMAGNETIC_LONGITUDE"]
):
    ds = read_dmsp(file_path)
    print(ds.LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP.values)
    data = {var: ds[var].values for var in variables}
    ds.close()
    return data

if __name__ == "__main__":
    read_dmsp()