import xarray as xr
from dataclasses import dataclass


@dataclass
class Period:
    time_min: str = "2010-10-19"
    time_max: str = "2010-11-19"

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        return ds.sel(time=slice(self.time_min, self.time_max))
