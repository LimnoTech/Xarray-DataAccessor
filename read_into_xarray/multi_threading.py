import warnings
from typing import Tuple


class DaskClass:
    """Prevents multiple clients from being started simultaneously"""
    dask_classes = []

    def __init__(
        self,
        thread_limit: int,
    ) -> None:

        # make sure a dask class is not already running
        DaskClass.dask_classes = [
            c for c in DaskClass.dask_classes if c.client.status == 'running'
        ]

        # close all but one if multiple are running
        if len(DaskClass.dask_classes) > 1:
            warnings.warn('Multiple dask clients were running!')
            for c in DaskClass.dask_classes[:-1]:
                c.close()

        # if one is running
        if len(DaskClass.dask_classes) > 0:
            dask_class = DaskClass.dask_classes[0]
            self.cluster = dask_class.cluster
            self.client = dask_class.client
            self.as_completed = dask_class.as_completed

        # controls it as threading, with one thread per N workers
        else:
            from dask.distributed import Client, LocalCluster, as_completed
            self.cluster = LocalCluster(
                n_workers=thread_limit,
                threads_per_worker=1,
                processes=False,
            )
            self.client = Client(self.cluster)
            self.as_completed = as_completed

            DaskClass.dask_classes.append(self)
        print(
            f'Dask dashboard URL: {self.client.scheduler_info()}'
        )


def get_multithread(
    use_dask: bool,
    thread_limit: int,
) -> Tuple[object, callable]:
    # multi process requests
    if use_dask:
        try:
            dask_class = DaskClass(thread_limit=thread_limit)
            client = dask_class.client

            # get as completed function
            as_completed_func = dask_class.as_completed
        except Exception as e:
            warnings.warn(
                f'Could not start dask -> reverting to concurrent.futures. '
                f'The following exception was received: {e}'
            )
            del as_completed
            use_dask = False

    if not use_dask:
        from concurrent.futures import (
            ThreadPoolExecutor,
            as_completed,
        )
        as_completed_func = as_completed
        client = ThreadPoolExecutor(max_workers=thread_limit)

    return (client, as_completed_func)
