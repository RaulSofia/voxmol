from rdkit import Chem
from collections import OrderedDict
from random_stuff import mol_list_to_sdf
from tqdm import tqdm


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
        for i, mol in tqdm(enumerate(mol_supplier), desc="Processing molecules"):
            if mol is not None:
                smiles_key = get_smiles_key(mol)
                if smiles_key not in index_dict:
                    index_dict[smiles_key] = []
                index_dict[smiles_key].append((mol_supplier, i))
    return index_dict


if __name__ == "__main__":
    sdf_files = [
        # './voxmol/dataset/data/drugs/raw/train_allconfs.sdf',
        './voxmol/dataset/data/drugs/raw/test_allconfs.sdf',
        './voxmol/dataset/data/drugs/raw/val_allconfs.sdf',
    ]
    mol_suppliers = [Chem.SDMolSupplier(sdf_file, removeHs=False) for sdf_file in sdf_files]

    index_dict = create_index_dict(mol_suppliers)
    print(f"Created index dictionary with {len(index_dict)} unique SMILES keys.")


