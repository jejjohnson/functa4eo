from dataclasses import dataclass
import xarray as xr


@dataclass
class Coarsen:
    latitude: int = 2
    longitude: int = 2
    boundary: str = "trim"

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.coarsen(
            latitude=self.latitude, longitude=self.longitude, boundary=self.boundary
        ).mean()
        return ds
