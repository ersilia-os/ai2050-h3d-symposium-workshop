from rdkit import Chem
from rdkit.Chem import Draw


def filter_valid_smiles(smiles_list):
    return [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(200, 200))

def draw_molecules_grid(smiles_list):
    mols = [Chem.MolFromSmiles(i) for i in smiles_list]
    return Draw.MolsToGridImage(mols)