import os
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity

import pandas as pd
import requests

from dotenv import load_dotenv

load_dotenv()

CHEMSPACE_API_KEY = os.getenv("CHEMSPACE_API_KEY")


class ChemSpaceSearch(object):
    """
        CSCS: Custom Request: Could be useful for requesting whole synthesis
        CSMB: Make-On-Demand Building Blocks
        CSSB: In-Stock Building Blocks
        CSSS: In-stock Screening Compounds
        CSMS: Make-On-Demand Screening Compounds
    """

    def __init__(self, chemspace_api_key: str = None):
        self.chemspace_api_key = chemspace_api_key
        self._renew_token()

    def _renew_token(self):
        self.chemspace_token = requests.get(
            url="https://api.chem-space.com/auth/token",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.chemspace_api_key}",
            },
        ).json()["access_token"]

    def _refactor(self, data):
        try:
            dfs = []
            # Convert this data into df
            for item in data["items"]:
                dfs_tmp = []
                smiles = item["smiles"]
                offers = item["offers"]

                for off in offers:
                    df_tmp = pd.DataFrame(off["prices"])
                    df_tmp["vendorName"] = off["vendorName"]
                    df_tmp["time"] = off["shipsWithin"]
                    df_tmp["purity"] = off["purity"]

                    dfs_tmp.append(df_tmp)

                df_this = pd.concat(dfs_tmp)
                df_this["smiles"] = smiles
                dfs.append(df_this)

            df = pd.concat(dfs).reset_index(drop=True)

            df["quantity"] = df["pack"].astype(str) + df["uom"]
            df["time"] = df["time"].astype(str) + " days"

            df = df.drop(columns=["pack", "uom"])
            df = df[df["priceUsd"].astype(str).str.isnumeric()]
            df = df[["priceUsd", "vendorName", "time", "purity", "quantity", "smiles"]]

            return df
        
        except:
            return None

    def _run(self, query, request_type="exact"):
        if request_type == "exact":
            count = 1
            categories = "CSMB,CSSB"
        elif request_type in ["sim", "sub"]:
            count = 5
            categories = "CSSS,CSMS"

        categories = "CSMB,CSSB,CSSS,CSMS"

        def _do_request(query, request_type, count, categories):
            data = requests.request(
                "POST",
                url=f"https://api.chem-space.com/v3/search/{request_type}?count={count}&page=1&categories={categories}",
                headers={
                    "Accept": "application/json; version=3.1",
                    "Authorization": f"Bearer {self.chemspace_token}",
                },
                data={"SMILES": f"{query}"},
            ).json()
            return data

        data = _do_request(query, request_type, count, categories)

        if "message" in data.keys():
            if data["message"] == "Your request was made with invalid credentials. Trying to renew token.":
                self._renew_token()

        data = _do_request(query, request_type, count, categories)
        data = self._refactor(data)
        if data is None:
            return None

        query_smiles = query
        all_smiles = list(set(data["smiles"].tolist()))

        def calculate_tanimoto_similarity(smiles_target, smiles_list):

            # Convert target SMILES to molecule
            mol_target = Chem.MolFromSmiles(smiles_target)
            
            # Generate fingerprints for the target molecule
            fp_target = AllChem.GetMorganFingerprintAsBitVect(mol_target, 2, nBits=2048)
            
            similarities = []
            for smiles in smiles_list:
                # Convert each SMILES to molecule
                mol = Chem.MolFromSmiles(smiles)
                
                # Generate fingerprints for the molecule
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                
                # Calculate Tanimoto similarity
                similarity = TanimotoSimilarity(fp_target, fp)
                similarities.append(similarity)
            
            return similarities

        similarities = calculate_tanimoto_similarity(query_smiles, all_smiles)
        sims = dict((k,v) for k,v in zip(all_smiles, similarities))
        all_smiles = data["smiles"].tolist()
        data["similarity"] = [sims[smi] for smi in all_smiles]

        return data
    
    def run(self, query):
        data = self._run(query, request_type="exact")
        if data is not None:
            return {"result": data, "request_type": "exact"}
        data = self._run(query, request_type="sim")
        return data


if __name__ == "__main__":
    cs = ChemSpaceSearch(chemspace_api_key=CHEMSPACE_API_KEY)
    smiles_string = "C1=CN=CXXX"
    print(cs.run(smiles_string))
