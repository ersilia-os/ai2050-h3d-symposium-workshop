import os
from chemcrow.agents import ChemCrow
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dotenv import load_dotenv
load_dotenv()


seed_smiles = "CCOC(=O)c1cnc2c(C)cc(C)cc2c1Nc1ccc(OC)c(OC)c1"
prompt = """
 Check if this molecule is available for purchase: {0}
 If not available, try to find similar molecules for purchase. In that case, give me a list of 5 similar molecules that are available for purchase.
 Give me the price of purchasable molecules
""".format(seed_smiles)


mrkl = ChemCrow(
    model="gpt-4",
    temp=0.1,
)

result = mrkl.run(prompt)





