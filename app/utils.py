import os
from rdkit import Chem
from rdkit.Chem import Draw
from dotenv import load_dotenv
from eosce.models import ErsiliaCompoundEmbeddings
from lol import LOL
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import requests
import json
from sklearn.ensemble import RandomForestClassifier


root = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(root, "..", ".env"))


def filter_valid_smiles(smiles_list):
    return [smiles for smiles in smiles_list if Chem.MolFromSmiles(smiles) is not None]


def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol, size=(200, 200))


def draw_molecules_grid(smiles_list):
    mols = [Chem.MolFromSmiles(i) for i in smiles_list]
    return Draw.MolsToGridImage(mols)


def query_nvidia_generative_chemistry(smiles, property="logP", minimize=False, minimum_similarity=0.85, num_molecules=30):

    if property=="logP":
        property = "plogP"

    invoke_url = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"

    headers = {
        "Authorization": "Bearer {0}".format(os.environ["NVIDIA_API_KEY"]),
        "Accept": "application/json",
    }

    payload = {
        "algorithm": "CMA-ES",
        "num_molecules": num_molecules,
        "property_name": property,
        "minimize": minimize,
        "min_similarity": minimum_similarity,
        "particles": 30,
        "iterations": 10,
        "smi": smiles
    }

    session = requests.Session()

    response = session.post(invoke_url, headers=headers, json=payload)

    response.raise_for_status()
    response_body = json.loads(response.json()["molecules"])
    data = [(r["sample"], r["score"]) for r in response_body]
    return data


def binarize_acinetobacter_data(data, cutoff):
    y = [1 if i <= cutoff else 0 for i in data["Mean"]]
    columns = list(data.columns)[:5]
    data = data[columns]
    data["Binary"] = y
    return data


def train_acinetobacter_ml_model(binary_data):
    embedder = ErsiliaCompoundEmbeddings()
    print("Calculating embeddings")
    X = embedder.transform(binary_data["SMILES"])
    print("Embeddings calculated")
    y = np.array(binary_data["Binary"])
    reducer = LOL(100)
    model = RandomForestClassifier()
    aurocs = []
    cv_data = []
    for i in range(5):
        print("CV iteration", i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train = reducer.fit_transform(X_train, y_train)
        X_test = reducer.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:,1]
        aurocs += [roc_auc_score(y_test, y_pred)]
        cv_data += [(y_test, y_pred)]
    print("Fitting final model")
    X = reducer.fit_transform(X, y)
    model.fit(X, y)
    results = {
        "reducer": reducer,
        "model": model,
        "aurocs": aurocs,
        "cv_data": cv_data,
        "X": X,
        "y": y
    }
    print(len(X))
    print(len(y))
    print(X[0])
    print("Done")
    return results

def train_final_acinetobacter_ml_model(binary_data):
    embedder = ErsiliaCompoundEmbeddings()
    X = embedder.transform(binary_data["SMILES"])
    y = np.array(binary_data["Binary"])
    reducer = LOL(100)
    model = RandomForestClassifier()
    X = reducer.fit_transform(X, y)
    model.fit(X, y)
    results = {
        "reducer": reducer,
        "model": model,
    }
    print("Done")
    return results

def predict_acinetobacter_ml_model(smiles_list, reducer, model):
    embedder = ErsiliaCompoundEmbeddings()
    print("Calculating embeddings")
    X = embedder.transform(smiles_list)
    print("Embeddings calculated")
    X = reducer.transform(X)
    y_pred = model.predict_proba(X)[:,1]
    return y_pred