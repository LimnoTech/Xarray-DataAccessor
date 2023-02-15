
class DaskClass:
    """Prevents multiple clients from being started simultaneously"""
    dask_classes = []

    def __init__(
        self,
        thread_limit: int,
    ) -> None:

        # make sure a dask class has not already be instantiated
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
