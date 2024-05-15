import dotenv
import os
from rxn4chemistry import RXN4ChemistryWrapper

dotenv.load_dotenv()

api_key = os.environ["RXN4CHEM_API_KEY"]
print(api_key)


rxn4chemistry_wrapper = RXN4ChemistryWrapper(api_key=api_key)

rxn4chemistry_wrapper.create_project('my_project')


response = rxn4chemistry_wrapper.predict_automatic_retrosynthesis(
    'Brc1c2ccccc2c(Br)c2ccccc12'
)
print(response)

import sys
sys.exit()
results = rxn4chemistry_wrapper.get_predict_automatic_retrosynthesis_results(
    response['prediction_id']
)
print(results['status'])
# NOTE: upon 'SUCCESS' you can inspect the predicted retrosynthetic paths.
print(results['retrosynthetic_paths'][0])