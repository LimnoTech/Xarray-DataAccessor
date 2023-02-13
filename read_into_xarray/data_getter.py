import multiprocessing
from typing import List
try:
    import dask.distributed
except ImportError:
    pass

# NOTE: maybe won't make sense since different threading schemes are ideal


class DataGetter:
    """Handles multiprocessing and the actual requests"""

    def dask_multiprocess(self):
        raise NotImplementedError

    def multiprocess(self):
        raise NotImplementedError
