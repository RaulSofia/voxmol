import gc
import os
import pickle
from time import sleep
from pympler import asizeof
import psutil


def get_mem():
    """
    Get the current memory usage of the process in bytes.

    Returns:
        int: The memory usage in bytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Resident Set Size


def open_pickled_data(file_path):
    """
    Open and load data from a pickled file.

    Args:
        file_path (str): The path to the pickled file.
    Returns:
        object: The data loaded from the pickled file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_data_mem_size(data):
    """
    Calculate the memory size of the given data object in bytes.

    Args:
        data (object): The data object to calculate the memory size for.

    Returns:
        int: The memory size of the data object in bytes.
    """
    return asizeof.asizeof(data)

if __name__ == "__main__":
    # Example usage
    start_mem = get_mem()
    data = open_pickled_data('./voxmol/dataset/data/drugs/raw/test_data.pickle')
    print(f"Loaded data with {type(data)}")
    print(f"Memory size of data: {get_data_mem_size(data)} bytes")
    end_mem = get_mem()
    print(f"Memory usage increased by {end_mem - start_mem} bytes")