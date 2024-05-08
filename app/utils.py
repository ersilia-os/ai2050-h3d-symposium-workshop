from rdkit import Chem


def filter_valid_smiles(smiles_list):
    return [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.Draw.MolToImage(mol, size=(200, 200))