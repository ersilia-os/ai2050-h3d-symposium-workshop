from ersilia_client import ErsiliaClient

ec = ErsiliaClient("https://eos9ei3-jvhi9.ondigitalocean.app/")

smiles_list = ["CCCCO", "CCCC"]

print(ec.run(smiles_list))