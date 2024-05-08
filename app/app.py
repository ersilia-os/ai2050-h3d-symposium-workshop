import os
import sys
import pandas as pd
import streamlit as st
from ersilia_client import ErsiliaClient
from rdkit.Chem import Draw
import random

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(root, "..", "data"))
sys.path.append(root)

from utils import filter_valid_smiles, draw_molecule, draw_molecules_grid

st.set_page_config(layout="wide", page_title='H3D Symposium AI Workshop', page_icon=':microbe:')

st.title(":microbe: AI for Antimicrobial Drug Discovery :pill:")

st.write("This app is part of the H3D Symposium (Livingstone, Zambia, 2024). Code for and more information can be found [here](https://github.com/ersilia-os/ai-intro-workshop).")

model_urls = {
    "eos80ch": "https://eos80ch-m365k.ondigitalocean.app/",
    "eos7yti": "https://eos7yti-thpl4.ondigitalocean.app/"
}

library_filenames = {
    "Global Health Box": "mmv_ghbox.csv",
    "Malaria Box": "mmv_malariabox.csv",
    "Pandemic Box": "mmv_pandemicbox.csv",
    "Pathogen Box": "mmv_pathogenbox.csv",
}

@st.cache_resource
def get_client(model_id):
    return ErsiliaClient(model_urls[model_id])

@st.cache_data
def read_library(library_filename):
    return list(pd.read_csv(os.path.join(data_dir, library_filename))["CAN_SMILES"])

clients = {model_id: get_client(model_id) for model_id in model_urls.keys()}

model_titles = {model_id: client.info["card"]["Title"] for model_id, client in clients.items()}

st.header("Library selection")

cols = st.columns(5)
selected_library = cols[0].radio("Select a screening library", list(library_filenames.keys()))
smiles_list = read_library(library_filenames[selected_library])

smiles_list = st.text_area("Enter a SMILES string", key="smiles", value=os.linesep.join(smiles_list)).split(os.linesep)
smiles_list = filter_valid_smiles(smiles_list)

cols[1].metric("Number of molecules", len(smiles_list))

sampled_smiles = random.sample(smiles_list, min(3, len(smiles_list)))
for i, smi in enumerate(sampled_smiles):
    cols[i+2].write("Sampled molecule {0}".format(i+1))

for i, smi in enumerate(sampled_smiles):
    cols[i+2].image(draw_molecule(smi))


st.header("Prioritization")

model_ids = st.multiselect("Select a model", list(model_titles.keys()), format_func=lambda x: model_titles[x])

def run_predictive_models(model_ids, smiles_list):
    return pd.DataFrame({"SMILES": smiles_list})
    results = {}
    for model_id in model_ids:
        client = clients[model_id]
        result = client.run(smiles_list)
        results[model_id] = result
    with st.status("Running Ersilia models..."):
        results = {}
        for i, model_id in enumerate(model_ids):
            st.write("Model {0}: {1}".format(model_id, model_titles[model_id]))
            client = clients[model_id]
            result = client.run(smiles_list)
            results[model_id] = result
    df = pd.DataFrame({"SMILES": smiles_list})
    for model_id, result in results.items():
        columns = list(result.columns)
        columns = [column for column in columns if column != "input"]
        df = pd.concat([df, result[columns]], axis=1)
    return df

dp = run_predictive_models(model_ids, smiles_list)

st.dataframe(dp)


st.header("Hit expansion")

seed_smiles = st.text_input("Enter a seed molecule SMILES string", value=dp.head(1)["SMILES"].values[0])

cols = st.columns(3)
cols[0].text("Seed molecule")
cols[0].image(draw_molecule(seed_smiles))

opt_property = cols[1].radio("Property to optimize", ["QED", "LogP", "Synthetic Accessibility"], index=0)

@st.cache_data
def run_generative_models(seed_smiles, opt_property, num_samples=100):
    return pd.DataFrame({"SMILES": [random.choice(smiles_list) for _ in range(num_samples)]})

dg = run_generative_models(seed_smiles, opt_property)

view = cols[2].radio("View", ["Table", "Images"])
st.dataframe(dg)

if view == "Images":
    image = draw_molecules_grid(dg["SMILES"])
    st.image(image)

