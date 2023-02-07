from typing import (
    Optional,
    Tuple,
    Union,
    List,
    Dict,
)
from pathlib import Path
from datetime import datetime
import xarray as xr
from prep_query import get_bounding_box

CoordsTuple = Tuple[float, float]


class GetERA5Data:
    """Main class to get a query data. Runs dask """

    def __init__(
        self,
        dataset_name: str,
        variables: Union[str, List[str]],
        start_time: Union[datetime, int],
        end_time: Union[datetime, int],
        dataset_source: Optional[str] = None,
        coordinates: Optional[Union[CoordsTuple, List[CoordsTuple]]] = None,
        csv_of_coords: Optional[Union[str, Path]] = None,
        shapefile: Optional[Union[str, Path]] = None,
        raster: Optional[Union[str, Path, xr.DataArray]] = None,
    ) -> None:

        # get bounding box coordinates
        inputs = {
            'coordinates': coordinates,
            'csv_of_coords': csv_of_coords,
            'shapefile': shapefile,
            'raster': raster,
        }

        valid_inputs = [(k, v) for k, v in inputs.items() if v is not None]

        if len(valid_inputs) == 0:
            raise ValueError(
                f'Must use one of the following AOI selectors: {list(inputs.keys())}'
            )
        elif len(valid_inputs) > 1:
            raise ValueError(
                f'Can only use one AOI selector! Multiple applied {inputs.items()}'
            )

        self.aoi_input = valid_inputs[0][1]
        self.aoi_input_type, self.aoi_input = valid_inputs[0]
        print(f'Using {self.aoi_input_type} to select AOI')

        self.bbox = self.get_bounding_box(
            aoi_input=self.aoi_input,
            aoi_input_type=self.aoi_input_type,
        )

    raise NotImplementedError
