import os
import sys
import pandas as pd
import streamlit as st
from ersilia_client import ErsiliaClient
import random

root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(root, "..", "data"))
sys.path.append(root)

from utils import filter_valid_smiles, draw_molecule

st.set_page_config(layout="wide")

st.title("AI for Antimicrobial Drug Discovery")

st.write("This is a web application that uses machine learning to predict the antimicrobial activity of a molecule.")

model_urls = {
    "eos80ch": "https://eos80ch-m365k.ondigitalocean.app/",
    "eos7yti": "https://eos7yti-thpl4.ondigitalocean.app/"
}

library_filenames = {
    "Global Health": "mmv_ghbox.csv",
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

cols = st.columns(5)
selected_library = cols[0].radio("Select a screening library", list(library_filenames.keys()))
smiles_list = read_library(library_filenames[selected_library])

smiles_list = st.text_area("Enter a SMILES string", key="smiles", value=os.linesep.join(smiles_list)).split(os.linesep)
smiles_list = filter_valid_smiles(smiles_list)

cols[1].metric("Number of molecules", len(smiles_list))

sampled_smiles = random.sample(smiles_list, 5)
cols[2].image(draw_molecule(sampled_smiles[0]))


model_ids = st.multiselect("Select a model", list(model_titles.keys()), format_func=lambda x: model_titles[x])


run_models = st.button("Run")

if run_models:

    with st.status("Running Ersilia models..."):
        results = {}
        for i, model_id in enumerate(model_ids):
            st.write("Model {0}: {1}".format(model_id, model_titles[model_id]))
            client = clients[model_id]
            result = client.run(smiles_list)
            results[model_id] = result


