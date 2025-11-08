import gc
import os
import pickle
from time import sleep
from pympler import asizeof
import psutil
from rdkit import Chem
from tqdm import tqdm


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

def mol_list_to_sdf(mol_list, sdf_path):
    """
    Save a list of RDKit molecule objects to an SDF file.

    Args:
        mol_list (list): List of RDKit molecule objects.
        sdf_path (str): The path to save the SDF file.
    """
    with Chem.SDWriter(sdf_path) as writer:
        for mol in tqdm(mol_list, desc="Writing SDF"):
            writer.write(mol)

def flatten_confs_geom_drugs(data, n_confs=5):
    """
    Flatten the Geom Drugs dataset to extract a specified number of conformers per molecule.
    If negative, like -1, extract all conformers.
    Funny how it doesn't double mem usage... passes by reference...

    Args:
        data (list): The original dataset containing tuples of (smiles, [conformers]).
        n_confs (int, optional): The number of conformers to extract per molecule. Defaults to 5.

    Returns:
        list: A flattened list of conformers (mol objects).
    """
    mols_confs = []
    for i, datum in enumerate(data):
        smiles, all_conformers = datum
        for j, conformer in enumerate(all_conformers):
            if n_confs > 0 and j >= n_confs:
                break
            mols_confs.append(conformer)
    return mols_confs

def get_data_mem_size(data):
    """
    Calculate the memory size of the given data object in bytes.

    Args:
        data (object): The data object to calculate the memory size for.

    Returns:
        int: The memory size of the data object in bytes.
    """
    return asizeof.asizeof(data)


def separate_sdf_by_split(input_sdf_path: str, output_dir: str, train_idxs: list, val_idxs: list, test_idxs: list):
    """
    Separate a large SDF file into train, validation, and test splits.

    Args:
        input_sdf_path (str): The path to the input SDF file.
        output_dir (str): The directory to save the separated SDF files.
        train_idxs (list): List of indices for the training set.
        val_idxs (list): List of indices for the validation set.
        test_idxs (list): List of indices for the test set.
    """
    mol_supplier = Chem.SDMolSupplier(input_sdf_path, removeHs=False, sanitize=False, strictParsing=False)
    train_writer = Chem.SDWriter(os.path.join(output_dir, "train.sdf"))
    val_writer = Chem.SDWriter(os.path.join(output_dir, "val.sdf")) 
    test_writer = Chem.SDWriter(os.path.join(output_dir, "test.sdf"))

    for i, mol in tqdm(enumerate(mol_supplier), total=len(mol_supplier), desc="Separating SDF by split"):
        if i in train_idxs:
            train_writer.write(mol)
        elif i in val_idxs:
            val_writer.write(mol)
        elif i in test_idxs:
            test_writer.write(mol)

    train_writer.close()
    val_writer.close()
    test_writer.close()


def get_indexes_from_csv_splits(train_csv_path: str, val_csv_path: str, test_csv_path: str):
    """
    Get the indexes of molecules from CSV split files.

    Args:
        train_csv_path (str): Path to the training CSV file.
        val_csv_path (str): Path to the validation CSV file.
        test_csv_path (str): Path to the test CSV file.
    Returns:
        tuple: Three lists containing the indexes for train, val, and test splits.
    """
    def read_indexes_from_csv(csv_path: str):
        indexes = []
        with open(csv_path, 'r') as f:
            next(f)  # Skip header
            for line in f:
                idx = int(line.strip().split(',')[0])
                indexes.append(idx)
        return indexes

    train_idxs = read_indexes_from_csv(train_csv_path)
    val_idxs = read_indexes_from_csv(val_csv_path)
    test_idxs = read_indexes_from_csv(test_csv_path)

    return train_idxs, val_idxs, test_idxs

if __name__ == "__main__":
    # Example usage
    # start_mem = get_mem()
    data = open_pickled_data('./voxmol/dataset/data/drugs/raw/train_data.pickle')
    data = flatten_confs_geom_drugs(data, n_confs=5)
    # for mol in data:
    #     print(type(mol), mol)
    # print(f"Loaded data with {type(data)}")
    # print(f"Memory size of data: {get_data_mem_size(data)} bytes")
    # end_mem = get_mem()
    # print(f"Memory usage increased by {end_mem - start_mem} bytes")
    mol_list_to_sdf(data, './voxmol/dataset/data/drugs/raw/train_5confs.sdf')



    # sdf_path = './voxmol/dataset/data/qm9/gdb9.sdf'
    # output_dir = './voxmol/dataset/data/qm9/'
    # train_idxs, val_idxs, test_idxs = get_indexes_from_csv_splits(
    #     os.path.join(output_dir, 'train.csv'),
    #     os.path.join(output_dir, 'val.csv'),
    #     os.path.join(output_dir, 'test.csv')
    # )
    # separate_sdf_by_split(sdf_path, output_dir, train_idxs, val_idxs, test_idxs)

