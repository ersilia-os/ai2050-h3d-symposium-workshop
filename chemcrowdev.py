import os
from chemcrow.agents import ChemCrow
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from dotenv import load_dotenv
load_dotenv()


seed_smiles = "CCOC(=O)c1cnc2c(C)cc(C)cc2c1Nc1ccc(OC)c(OC)c1"
prompt = """
 Synthesize this molecule, and give me the steps: {0}
 Give me the IUPAC name, its functional groups, and its likely properties. Also, tell me what chemotypes are present in this molecule.
 Explain this molecule so that a chemist can understand it.
""".format(seed_smiles)


mrkl = ChemCrow(
    model="gpt-4",
    temp=0.1,
)

result = mrkl.run(prompt)





