import polars as pl
from time import time
import glob
import numpy as np
import polars as pl
import scipy.io as scio
from multiprocessing import Pool
from itertools import repeat
import sys


def extract_relevant_columns(data: dict) -> dict:
    """Remove first 3 columns since they are not unique to the data"""
    return {
        col_name: data[col_name] for col_name in list(data.keys())[3:]
    }


def unnest_dict_values(data: dict) -> dict:
    """Removes nesting, flattening the dict"""
    # create dictionary with unnrested array and sampling rate in list
    return {
        col_name: [
            data[col_name]["Rate"][0][0][0][0],
            data[col_name]["data"][0][0].flatten()
        ] for col_name in data
    }


def up_sample_array(
        array: np.array,
        rate: int,
        max_rate: int = 16
) -> np.array:
    """
    Resamples an array to the max rate found in dataset by
    interpolating samples using previous real value until next sample.

    Args:
        array: array to interpolate
        rate: sampling rate of variable
        max_rate: highest sampling rate of all variables

    Returns:
        resampled_array
    """
    # don't upsample if already at max rate
    if rate == max_rate:
        return array
    # empty array to fill up
    sample_factor = int(max_rate / rate)
    output_array_length = int(len(array) * sample_factor)
    resampled_array = np.empty(
        shape=(output_array_length,),
        dtype=np.float32
    )
    # loop every Nsample_factor element and each element of input array
    for i, val_index in zip(
            range(0, output_array_length - sample_factor, sample_factor),
            range(len(array))
    ):
        # fill every Nsample_factor samples within each loop iteration
        resampled_array[i:i + sample_factor] = np.repeat(array[val_index], sample_factor)

    return resampled_array


def interpolate_data(data: dict) -> dict:
    """Applies interpolation to each variable in data"""
    return {
        col: up_sample_array(data[col][1], data[col][0]) for col in data
    }


# gather list of files in directory
def get_filepaths(directory: str) -> list:
    """
    Retrieves folder/filepaths in directory recursively.
    """
    return glob.glob(directory + '/**', recursive=True)


def read_file(filename: str) -> dict:
    """Reads file and returns flat data"""
    data = scio.loadmat(filename)
    data = extract_relevant_columns(data)
    data = unnest_dict_values(data)

    return data


def process_file(data: dict) -> pl.DataFrame:
    """Returns data in flat dict to interpolated polars dataframe"""
    data = interpolate_data(data)
    # create polars dataframe
    df = pl.DataFrame(data)
    # note there are no text columns
    # keeping all vars float 32 stops a downstream bug
    df = df.cast(pl.Float32)

    return df


def format_filename(filename: str) -> str:
    """Formats filename by stripping out folder structure"""
    filename = filename.split('/')[-1]
    filename = filename.split('.mat')[0]

    return filename


def write_parquet(
        df: pl.DataFrame,
        filename: str,
        write_dir: str
):
    """
    Writes polars dataframe to parquet file.
    Current implementation supports only writing
    to local disk.

    Args:
        df: polars dataframe
        filename: filename for output file
        write_dir: directory to write file
    """
    # create pl df and write to parquet
    filename = format_filename(filename)
    path = f"{write_dir}/{filename}.parquet"
    df.write_parquet(path)


def mat2pq(
        filename: str,
        parq_dir: str,
        progress: str
):
    """
    Pipeline function that converts a single .mat files to
    a single parquet file. Progress string included so can
    track progress in terms of total files in the terminal.
    """
    data = read_file(filename)
    df = process_file(data)
    write_parquet(df, filename, parq_dir)
    # execution with pool is not ordered so results may occur in stange order
    print(f'progress writing parquet files = {progress}', flush=True)


def build_progress_list(all_filepaths: list) -> list:
    """
    Returns list of file number out of total files
    used to track program progress
    """
    total_files = len(all_filepaths)
    file_numbers = range(1, total_files, 1)

    return [
        str(file_number) + '/' + str(total_files) for file_number in file_numbers
    ]


def convert_matlab_to_parquet(
        mat_dir: str,
        parq_dir: str

):
    """
    Takes each .mat file in mat dir and reads the data
    interpolates arrays to each variable has the same number
    of values so that the data can become tabular.
    Writes to parquet files in parq_dir.
    Uses multiprocessing.

    Args:
        mat_dir: directory holding .mat files (recursive)
        parq_dir: directory where to write parquets files (must exist)
    """
    # store start time in order to find total elapsed at end
    start_time = time()
    # gather paths of all .mat files
    all_filepaths = get_filepaths(mat_dir)
    # setting up progress printer
    progress_list = build_progress_list(all_filepaths)
    # use multiprocessing to allow each cpu thread to convert
    # a .mat files to .parquet in parallel
    with Pool() as pool:
        pool.starmap_async(
            mat2pq,
            zip(all_filepaths, repeat(parq_dir), progress_list)
        )
        pool.close()
        pool.join()
    # calculate total time taken for program to run
    end_time = time()
    elapsed = (end_time - start_time) / 3600
    print(f'total time elasped = {elapsed} hours')


if __name__ == '__main__':
    convert_matlab_to_parquet(mat_dir=sys.argv[0], parq_dir=sys.argv[1])
    exit()