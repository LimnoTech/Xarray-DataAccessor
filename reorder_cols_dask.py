import pandas as pd
from dask.distributed import Client
import dask.dataframe as dd
from pathlib import Path


def main(
    data_directory: Path,
    tag: str,
) -> None:
    # fire up dask distributed
    client = Client()
    cluster = client.cluster

    # find files to process
    in_files = []
    for file in data_directory.iterdir():
        if tag in file.name and file.suffix == '.csv':
            in_files.append(file)
    print(f'Processing files: {in_files}')

    # process the files
    for file in in_files:
        # read into dask
        print(f'Processing {file}')
        in_df = dd.read_csv(
            file,
            parse_dates=['datetime'],
        )

        # reorder the integer columns and turn back to strings
        int_cols = in_df.columns[1:].astype('int').sort_values()
        reordered_ints = int_cols.astype('str')
        del int_cols

        # use Index.append() to add datetime back to the start
        reordered_index = pd.Index(['datetime']).append(reordered_ints)

        # make function we can map to partitions - hardcode in reordered_index
        def reindex_cols(partition):
            return partition.reindex(columns=reordered_index)

        # reorder distributed!
        out_df = in_df.map_partitions(reindex_cols)
        out_csv = out_df.to_csv(file.replace(tag, ''))
        print(f'CSV saved @ {out_csv}')

    # shut down client/cluster
    client.close()
    cluster.close()


if __name__ == '__main__':
    # find files in the data directory
    MET_DATA_DIR = Path('X:\AAOWorking\LEEM2\Met_Data')
    TAG = '_old'
    main(MET_DATA_DIR, TAG)
