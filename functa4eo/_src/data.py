from dataclasses import dataclass
import xarray as xr
from pathlib import Path


@dataclass
class MEDTemperatureDataset:
    variable_name: str = "thetao"
    filename: str = "MED_GLORYS_TEMPERATURE_DAILY.nc"

    def open_dataset(self, base_path):
        filepath = Path(base_path).joinpath(self.filename)
        return xr.open_mfdataset(
            filepath, preprocess=self.correct_units, engine="netcdf4"
        )

    def correct_units(self, ds):
        ds[self.variable_name].attrs["units"] = "degC"
        return ds


@dataclass
class MEDSSHDataset:
    variable_name: str = "zos"
    filename: str = "MED_GLORYS_SSH_DAILY.nc"

    def open_dataset(self, base_path):
        filepath = Path(base_path).joinpath(self.filename)
        return xr.open_mfdataset(
            filepath, preprocess=self.correct_units, engine="netcdf4"
        )

    def correct_units(self, ds):
        ds[self.variable_name].attrs["units"] = "meters"
        return ds


@dataclass
class MEDUVelDataset:
    variable_name: str = "uo"
    filename: str = "MED_GLORYS_SSH_DAILY.nc"

    def open_dataset(self, base_path):
        filepath = Path(base_path).joinpath(self.filename)
        return xr.open_mfdataset(
            filepath, preprocess=self.correct_units, engine="netcdf4"
        )

    def correct_units(self, ds):
        ds[self.variable_name].attrs["units"] = "meter / second"
        return ds


@dataclass
class MEDVVelDataset:
    variable_name: str = "vo"
    filename: str = "MED_GLORYS_SSH_DAILY.nc"

    def open_dataset(self, base_path):
        filepath = Path(base_path).joinpath(self.filename)
        return xr.open_mfdataset(
            filepath, preprocess=self.correct_units, engine="netcdf4"
        )

    def correct_units(self, ds):
        ds[self.variable_name].attrs["units"] = "meter / second"
        return ds


def _open_and_validate(path):
    return xr.open_mfdataset(path, preprocess=validate_units, engine="netcdf4")


def validate_units(ds):
    var_names = list(ds.data_vars.keys())

    if "zos" in var_names:
        ds["zos"].attrs["units"] = "meters"
    if "thetao" in var_names:
        ds["thetao"].attrs["units"] = "degC"
    if "bottomT" in var_names:
        ds["bottomT"].attrs["units"] = "degC"
    if "uo" in var_names:
        ds["uo"].attrs["units"] = "meter / second"
    if "vo" in var_names:
        ds["vo"].attrs["units"] = "meter / second"
    return ds
