import warnings
from typing import Tuple, Optional


class DaskClass:
    """Prevents multiple clients from being started simultaneously"""
    dask_classes = []

    def __init__(
        self,
        n_workers: int,
        threads_per_worker: Optional[int] = None,
        processes: bool = True,
        close_existing_client: bool = False,
    ) -> None:

        # make sure a dask class is not already running
        from dask.distributed import Client, LocalCluster, as_completed

        # close all but one if multiple are running
        if len(DaskClass.dask_classes) > 1:
            warnings.warn('Multiple dask clients were running!')
            for c in DaskClass.dask_classes[:-1]:
                c.client.close()
                c.cluster.close()

        # force closure if we want to change parameters
        if close_existing_client and len(DaskClass.dask_classes) > 0:
            DaskClass.dask_classes[-1].client.close()
            DaskClass.dask_classes[-1].cluster.close()
            DaskClass.dask_classes = []

        # if one is running
        if len(DaskClass.dask_classes) > 0:
            dask_class = DaskClass.dask_classes[-1]
            self.cluster = dask_class.cluster
            self.client = Client(self.cluster)
            self.as_completed = dask_class.as_completed

        # controls it as threading, with one thread per N workers
        else:
            self.cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=processes,
                dashboard_address=':8787',
            )
            self.client = Client(self.cluster)
            self.as_completed = as_completed

            DaskClass.dask_classes.append(self)


def get_multithread(
    use_dask: bool,
    n_workers: int,
    threads_per_worker: int,
    processes: bool = True,
    close_existing_client: bool = False,
) -> Tuple[object, callable]:
    """Give you a multithread/process executer.

    Arguments:
        use_dask: Whether to default to using dask.
        n_workers: # of processes to run.
        threads_per_worker: # of threads to run per worker.
            NOTE: Only applied when processes=True.
        processes: Whether to default to multiprocessing (True) or multithreading (False).
        close_existing_client: Closes existing dask client. For when you want to switch settings.

    Returns:
        A tuple with your executer [0], and the as_completed() function [1].
    """
    if use_dask:
        try:
            dask_class = DaskClass(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=processes,
                close_existing_client=close_existing_client,
            )
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
        if processes:
            from concurrent.futures import (
                ProcessPoolExecutor,
                as_completed,
            )
            client = ProcessPoolExecutor(
                max_workers=n_workers,
                max_tasks_per_child=threads_per_worker,
            )
        else:
            from concurrent.futures import (
                ThreadPoolExecutor,
                as_completed,
            )
            client = ThreadPoolExecutor(max_workers=n_workers)
        as_completed_func = as_completed

    return (client, as_completed_func)
