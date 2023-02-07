from datetime import datetime
from typing import List, Dict
import xarray as xr


class AWSDataAccessor:
    def get_data(
        self,
        dataset_name: str,
        variables: List[str],
        start_dt: datetime,
        end_dt: datetime,
        bbox: Dict[str, float],
        multithread: bool = True,
    ) -> xr.Dataset:
        raise NotImplementedError
