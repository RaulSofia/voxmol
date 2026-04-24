import gc
import csv
import os
import pickle
from time import sleep
from collections import OrderedDict
from collections.abc import Mapping
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

    The loaded object is normalized into an OrderedDict keyed by canonical,
    isomeric SMILES with isotopes removed.

    Args:
        file_path (str): The path to the pickled file.
    Returns:
        OrderedDict: The normalized data loaded from the pickled file.
    """
    print(f"[load] reading pickle: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"[load] normalizing pickle: {file_path}")
    return normalize_pickled_data(data)


def remove_isotopes(smiles):
    """
    Remove isotopic labels from a SMILES string while preserving stereo.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        str: Canonical isomeric SMILES without isotopes.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    for atom in mol.GetAtoms():
        atom.SetIsotope(0)

    mol = Chem.AddHs(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def ensure_explicit_hydrogens(mol):
    """
    Return a copy of a molecule with all hydrogens made explicit.

    Args:
        mol: RDKit molecule.

    Returns:
        RDKit molecule: A copy with explicit hydrogens.
    """
    return Chem.AddHs(Chem.Mol(mol), addCoords=True)


def canonicalize_smiles(smiles):
    """
    Canonicalize a SMILES string after removing isotopic information.

    Args:
        smiles (str): Input SMILES string.

    Returns:
        str: Canonical isomeric SMILES without isotopes.
    """
    return remove_isotopes(smiles)


def canonicalize_molecule_smiles(mol):
    """
    Canonicalize a molecule into an explicit-hydrogen SMILES string.

    Args:
        mol: RDKit molecule.

    Returns:
        str: Canonical isomeric SMILES with isotopes removed and hydrogens explicit.
    """
    normalized_mol = Chem.Mol(mol)
    for atom in normalized_mol.GetAtoms():
        atom.SetIsotope(0)

    normalized_mol = Chem.AddHs(normalized_mol, addCoords=True)
    return Chem.MolToSmiles(normalized_mol, canonical=True, isomericSmiles=True)


def normalize_pickled_data(data):
    """
    Normalize loaded molecule data into an OrderedDict keyed by SMILES.

    Args:
        data: Legacy list/tuple data or a mapping of SMILES to conformers.

    Returns:
        OrderedDict: Ordered mapping from canonical SMILES to conformer lists.
    """
    normalized_data = OrderedDict()

    if isinstance(data, Mapping):
        items = data.items()
    else:
        items = data

    for smiles, conformers in tqdm(items, desc="Normalizing SMILES groups"):
        canonical_smiles = canonicalize_smiles(smiles)
        explicit_conformers = [ensure_explicit_hydrogens(conformer) for conformer in conformers]
        normalized_data.setdefault(canonical_smiles, []).extend(explicit_conformers)

    return normalized_data


def merge_pickled_data_objects(data_objects):
    """
    Merge multiple normalized data objects into one OrderedDict.

    Args:
        data_objects (list): List of normalized data objects.

    Returns:
        OrderedDict: Combined ordered mapping grouped by canonical SMILES.
    """
    merged_data = OrderedDict()

    for data in tqdm(data_objects, desc="Merging normalized data objects"):
        normalized_data = normalize_pickled_data(data)
        for smiles, conformers in normalized_data.items():
            merged_data.setdefault(smiles, []).extend(conformers)

    return merged_data


def load_and_merge_pickled_files(file_paths):
    """
    Load multiple pickle files and merge them into one normalized OrderedDict.

    Args:
        file_paths (list[str]): Pickle file paths.

    Returns:
        OrderedDict: Combined ordered mapping grouped by canonical SMILES.
    """
    merged_data = OrderedDict()

    for file_path in tqdm(file_paths, desc="Loading pickle files"):
        data = open_pickled_data(file_path)
        print(f"[merge] {file_path}: {len(data)} smiles groups")
        for smiles, conformers in data.items():
            merged_data.setdefault(smiles, []).extend(conformers)

    # Re-normalize at the end to guarantee key format and molecule consistency.
    print(f"[merge] re-normalizing merged data with {len(merged_data)} smiles groups")
    return normalize_pickled_data(merged_data)


def load_and_merge_sdf_files(file_paths):
    """
    Load multiple SDF files and merge them into one normalized OrderedDict.

    Each molecule is normalized to isotope-free explicit-H form and grouped by
    canonical isomeric explicit-H SMILES.

    Args:
        file_paths (list[str]): SDF file paths.

    Returns:
        OrderedDict: Combined ordered mapping grouped by canonical SMILES.
    """
    merged_data = OrderedDict()

    for file_path in file_paths:
        mol_supplier = Chem.SDMolSupplier(
            file_path,
            removeHs=False,
            sanitize=False,
            strictParsing=False,
        )

        for mol in tqdm(mol_supplier, desc=f"Reading {os.path.basename(file_path)}"):
            if mol is None:
                continue

            normalized_mol = Chem.Mol(mol)
            for atom in normalized_mol.GetAtoms():
                atom.SetIsotope(0)

            normalized_mol = ensure_explicit_hydrogens(normalized_mol)
            smiles = canonicalize_molecule_smiles(normalized_mol)
            merged_data.setdefault(smiles, []).append(normalized_mol)

    return merged_data


def cap_conformers_per_smiles(data, max_conformers_per_smiles):
    """
    Cap the number of conformers stored under each SMILES key.

    Args:
        data (OrderedDict): Ordered mapping from SMILES to conformer lists.
        max_conformers_per_smiles (int): Max conformers per key. Negative keeps all.

    Returns:
        OrderedDict: Ordered mapping with capped conformers per key.
    """
    print(f"[cap] limiting conformers per smiles to {max_conformers_per_smiles}")
    if max_conformers_per_smiles < 0:
        return OrderedDict((smiles, list(conformers)) for smiles, conformers in tqdm(data.items(), desc="Copying conformer groups"))

    capped_data = OrderedDict()
    for smiles, conformers in tqdm(data.items(), desc="Capping conformers per smiles"):
        capped_data[smiles] = list(conformers[:max_conformers_per_smiles])

    return capped_data


def merge_and_flatten_confs_geom_drugs(data_objects, n_confs=5):
    """
    Merge multiple data objects and flatten their conformers in order.

    Args:
        data_objects (list): List of normalized data objects.
        n_confs (int, optional): The number of conformers to extract per molecule.

    Returns:
        list: A flattened list of conformers (mol objects).
    """
    merged_data = merge_pickled_data_objects(data_objects)
    return flatten_confs_geom_drugs(merged_data, n_confs=n_confs)

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
    Flatten an ordered molecule mapping into a list of conformers.

    If n_confs is negative, all conformers are returned for each SMILES key.

    Args:
        data (OrderedDict): Ordered mapping from SMILES to lists of conformers.
        n_confs (int, optional): The number of conformers to extract per molecule. Defaults to 5.

    Returns:
        list: A flattened list of conformers (mol objects).
    """
    mols_confs = []
    for smiles, all_conformers in tqdm(data.items(), desc="Flattening conformers by smiles"):
        for j, conformer in enumerate(tqdm(all_conformers, desc=f"{smiles}", leave=False)):
            if n_confs > 0 and j >= n_confs:
                break
            mols_confs.append(conformer)
    return mols_confs


def smiles_to_csv(data, csv_path):
    """
    Write ordered SMILES data to a CSV file.

    Args:
        data (OrderedDict): Ordered mapping from SMILES to lists of conformers.
        csv_path (str): Destination CSV path.
    """
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['smiles', 'num_conformers'])
        for smiles, conformers in tqdm(data.items(), desc="Writing smiles CSV"):
            writer.writerow([smiles, len(conformers)])


def conformer_list_to_smiles_txt(conformers, txt_path, include_index=True):
    """
    Write one canonical explicit-H SMILES per conformer to a TXT file.

    SMILES are extracted directly from each conformer, so line order matches the
    conformer list order exactly (and therefore can match SDF record order).

    Args:
        conformers (list): Flat list of RDKit conformer molecules.
        txt_path (str): Destination TXT path.
        include_index (bool): Whether to prefix each line with the conformer index.
    """
    print(f"[write] writing smiles txt: {txt_path}")
    with open(txt_path, 'w') as txt_file:
        for i, conformer in tqdm(list(enumerate(conformers)), desc="Writing smiles txt"):
            smiles = canonicalize_molecule_smiles(conformer)
            if include_index:
                txt_file.write(f"{i}\t{smiles}\n")
            else:
                txt_file.write(f"{smiles}\n")


def validate_smiles_key_consistency(data, raise_on_mismatch=False, max_examples=10):
    """
    Check whether each conformer in an OrderedDict matches the SMILES key.

    Args:
        data (OrderedDict): Ordered mapping from SMILES to lists of conformers.
        raise_on_mismatch (bool, optional): Raise a ValueError if mismatches are found.
        max_examples (int, optional): Maximum number of mismatches to include in the report.

    Returns:
        dict: Validation summary with counts and example mismatches.
    """
    print("[validate] checking smiles-key consistency")
    mismatches = []
    total_conformers = 0

    for key_smiles, conformers in tqdm(data.items(), desc="Validating smiles groups"):
        for conformer_index, conformer in enumerate(tqdm(conformers, desc=f"{key_smiles}", leave=False)):
            total_conformers += 1
            conformer_smiles = canonicalize_molecule_smiles(conformer)
            if conformer_smiles != key_smiles:
                mismatches.append(
                    {
                        "key_smiles": key_smiles,
                        "conformer_smiles": conformer_smiles,
                        "conformer_index": conformer_index,
                    }
                )

    validation_result = {
        "is_valid": len(mismatches) == 0,
        "num_keys": len(data),
        "num_conformers": total_conformers,
        "num_mismatches": len(mismatches),
        "mismatches": mismatches[:max_examples],
    }

    if raise_on_mismatch and mismatches:
        raise ValueError(
            "Found conformers whose explicit-hydrogen canonical SMILES do not match their key: "
            f"{mismatches[:max_examples]}"
        )

    return validation_result

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
    # Previous example usage (kept for reference):
    # start_mem = get_mem()
    # data = open_pickled_data('./voxmol/dataset/data/drugs/raw/train_data.pickle')
    # data = flatten_confs_geom_drugs(data, n_confs=5)
    # for mol in data:
    #     print(type(mol), mol)
    # print(f"Loaded data with {type(data)}")
    # print(f"Memory size of data: {get_data_mem_size(data)} bytes")
    # end_mem = get_mem()
    # print(f"Memory usage increased by {end_mem - start_mem} bytes")
    # mol_list_to_sdf(data, './voxmol/dataset/data/drugs/raw/train_5confs.sdf')

    input_pickle_files = [
        './voxmol/dataset/data/drugs/raw/val_data.pickle',
        './voxmol/dataset/data/drugs/raw/test_data.pickle',
        './voxmol/dataset/data/drugs/raw/train_data.pickle',
    ]

    # Previous SDF-based input list (kept for reference):
    # input_sdf_files = [
    #     './voxmol/dataset/data/drugs/raw/train_allconfs.sdf',
    #     './voxmol/dataset/data/drugs/raw/val.sdf',
    #     './voxmol/dataset/data/drugs/raw/test.sdf',
    # ]

    output_dir = './voxmol/dataset/data/drugs/raw/'
    output_sdf_path = os.path.join(output_dir, 'geom_drugs.sdf')
    output_txt_path = os.path.join(output_dir, 'geom_drugs.smi')
    max_conformers_per_smiles = -1

    # Previous pickle merge call (kept for reference):
    # merged_data = load_and_merge_pickled_files(input_pickle_files)

    merged_data = load_and_merge_pickled_files(input_pickle_files)
    print("loaded and merged data from pickle files")

    # Optional strict consistency check before capping/export.
    validation = validate_smiles_key_consistency(merged_data)
    print(f"Validation: {validation['num_mismatches']} mismatches over {validation['num_conformers']} conformers")

    capped_data = cap_conformers_per_smiles(merged_data, max_conformers_per_smiles)
    print(f"[stage] capped data contains {len(capped_data)} smiles groups")
    flat_conformers = flatten_confs_geom_drugs(capped_data, n_confs=max_conformers_per_smiles)
    print(f"[stage] flattened to {len(flat_conformers)} conformers")

    mol_list_to_sdf(flat_conformers, output_sdf_path)
    conformer_list_to_smiles_txt(flat_conformers, output_txt_path, include_index=True)
    print(f"Wrote {len(flat_conformers)} conformers to {output_sdf_path}")
    print(f"Wrote aligned SMILES TXT to {output_txt_path}")



    # sdf_path = './voxmol/dataset/data/qm9/gdb9.sdf'
    # output_dir = './voxmol/dataset/data/qm9/'
    # train_idxs, val_idxs, test_idxs = get_indexes_from_csv_splits(
    #     os.path.join(output_dir, 'train.csv'),
    #     os.path.join(output_dir, 'val.csv'),
    #     os.path.join(output_dir, 'test.csv')
    # )
    # separate_sdf_by_split(sdf_path, output_dir, train_idxs, val_idxs, test_idxs)

