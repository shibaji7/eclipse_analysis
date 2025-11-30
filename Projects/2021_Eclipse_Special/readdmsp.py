from xarray import open_dataset
import datetime as dt

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

def get_energy_bins(file_path=None):
    import aacgmv2
    ds = read_dmsp(file_path)
    energy_bins = ds["ENERGY_FLUX_SOUTH_MAP"]
    mlat, mlon = (
        ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values,
        ds["LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP"].values
    )
    ds.close()
    print(ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values)
    # glat, glon = aacgmv2.convert_latlon(
    #     mlat, mlon, 300, dt.datetime(2021, 12, 4, 7),
    #     method_code="A2G"
    # )
    # return energy_bins, glat, glon

if __name__ == "__main__":
    # read_dmsp()
    get_energy_bins()