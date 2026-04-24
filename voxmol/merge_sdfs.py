from rdkit import Chem
from collections import OrderedDict
from random_stuff import mol_list_to_sdf
from tqdm import tqdm
import csv


def get_smiles_key(mol):
    # Generate a canonical SMILES with explicit hydrogens for the given molecule
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)  # Remove isotope information to ensure consistency
    return Chem.MolToSmiles(
        mol, 
        canonical=True,        # Ensures the SMILES is canonical (default is True)
        isomericSmiles=True,   # Distinguishes stereochemistry/isomers (default is True)
        allHsExplicit=True     # Forces every hydrogen to be written out as an exact atom (e.g., [H])
    )

def create_index_dict(mol_supplier_list: list):
    index_dict = OrderedDict()
    for mol_supplier in mol_supplier_list:
        for i, mol in tqdm(enumerate(mol_supplier), desc="Processing molecules", total=len(mol_supplier)):
            if mol is not None:
                smiles_key = get_smiles_key(mol)
                if smiles_key not in index_dict:
                    index_dict[smiles_key] = []
                index_dict[smiles_key].append((mol_supplier, i))
    return index_dict


if __name__ == "__main__":
    sdf_files = [
        './voxmol/dataset/data/drugs/raw/train_allconfs.sdf',
        './voxmol/dataset/data/drugs/raw/test_allconfs.sdf',
        './voxmol/dataset/data/drugs/raw/val_allconfs.sdf',
        # './voxmol/dataset/data/qm9/val.sdf',
        # './voxmol/dataset/data/qm9/test.sdf',
        # './voxmol/dataset/data/qm9/train.sdf',
    ]

    # MAX_CONFORMER_COUNT = None # Set to None to keep all conformers, or set to an integer to limit the number of conformers per unique SMILES
    MAX_CONFORMER_COUNT = 5


    mol_suppliers = [Chem.SDMolSupplier(sdf_file, removeHs=False) for sdf_file in sdf_files]

    index_dict = create_index_dict(mol_suppliers)
    print(f"Created index dictionary with {len(index_dict)} unique SMILES keys.")

    if MAX_CONFORMER_COUNT is not None:
        for smiles_key in tqdm(index_dict, desc="SMILES keys and counts"):
            index_dict[smiles_key] = index_dict[smiles_key][:MAX_CONFORMER_COUNT]

    with open('./voxmol/dataset/data/drugs/raw/geom_drugs_5confs.smi', 'w') as smi_file:
        with Chem.SDWriter('./voxmol/dataset/data/drugs/raw/geom_drugs_5confs.sdf') as sdf_writer:
            for smiles_key, mol_locations in tqdm(index_dict.items(), desc="Processing SMILES keys"):
                for mol_supplier, idx in mol_locations:
                    mol = mol_supplier[idx]
                    if mol is not None:
                        sdf_writer.write(mol)
                        smi_file.write(f"{smiles_key}\n")

    count_dict = OrderedDict((key, len(locations)) for key, locations in index_dict.items())
    

    csv_file_path = './voxmol/dataset/data/drugs/raw/geom_drugs_5confs_counts.csv'
    
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(['SMILES', 'Count'])
        
        for smiles_key, count in count_dict.items():
            writer.writerow([smiles_key, count])
            
    print(f"Saved counts to {csv_file_path}")
