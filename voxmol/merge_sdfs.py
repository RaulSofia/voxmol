from rdkit import Chem
from collections import OrderedDict
from random_stuff import mol_list_to_sdf, remove_isotopes, ensure_explicit_hydrogens


def create_index_dict(mol_supplier_list):
    index_dict = OrderedDict()
    for mol in mol_supplier_list:
        if mol is not None:
            name = mol.GetProp('_Name')
            index_dict[name] = mol
    return index_dict