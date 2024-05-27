# Introductory workshop to AI/ML for antimicrobial drug discovery

This repository contains the materials for an introduction to AI tools for drug discovery, delivered at the 5th H3D Symposium in Livingstone, Zambia. The demo contained within this repository is only brought alive online during the workshop delivery. If you wish to use it for demo purposes, please email us at hello[at]ersilia[dot]io

## Hands on Activity
You can find detailed information about the proposed demo in this [page](https://ersilia.gitbook.io/ersilia-book/training-materials/ai-antimicrobial-dd). Find below a quick summary!

The participant is presented with the following problem statement: _We are a laboratory specialised in antimicrobial research. We have received a library of compounds from a collaborator and we need to explore it and identify the best candidates._

Some of the considerations to take into account:
* We can synthesise compounds but our throughput is 50 compounds per month
* We have a tight timeline (2 months) to provide the results
* Our collaborators are keen on exploring analogues of the molecules in their library as well
* The selected compounds need to meet the following criteria: 
  * Synthetically accessible in the laboratory 
  * Good ADME profile 
  * High activity against at least one pathogen in the WHO priority list

### Step 0: AMA
A simple plug to GPT3.5 to ask questions around pathogens.

### Step 1: Train an ML Model
Train a simple classifier model based on the dataset for _A.baumannii_ activity reported in Liu et al, 2023. The participants can play with different cut-offs and the demo performs an automated 5-fold cross-validation.

### Step 2: Prioritize candidates
Participants will use a given dataset (prepared from ChEMBL) and run predictions using the just trained _A.baumannii_ model as well as two models from the Ersilia Model Hub: Synthetic Accessibility Score (Ertl et al, 2009) and hERG cardiotoxicity (Jiménez-Luna et al, 2021). The goal is to select the best molecule according to the predicted values (high activity against _A.baumannii_, good synthetic accessibility and low cardiotoxicity).

### Step 3: Generate new candidates
Using the best candidate, try the MolMIM generator (Reidenbach et al, 2022) to obtain analogues with better drug-likeness (QED and LogP).

### Step 4: Purchase molecules
We use a quick search on Chem-Space to find which compounds are directly purchasable.

## Requirements
Requirements are listed in the Dockerfile. To run it locally:
1. Create a conda environment and install the required packages
2. Clone the repository
3. From the root of the repository, run `streamlit run app/app.py`

To run the app, the models will need to be deployed online/locally. The URL's for the models can be modified in the `info.py` file.
In addition, we use a number of services that require an API (all of them offer free credits that should be sufficient for the usage in this demo). Please create an `.env` file with the required API Keys.

## References
The models used in this demo come from:
* Liu et al. Deep learning-guided discovery of an antibiotic targeting Acinetobacter baumannii. Nature Chemical Biology, 2023
* Ertl & Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Chemioinformatics, 2009
* Jiménez-Luna et al. Coloring Molecules with Explainable Artificial Intelligence for Preclinical Relevance Assessment. Journal of Chemical Information and Modelling, 2021
* Reidenbach et al. Improving small molecule generation using mutual information machine. ArXiv. 2022

## Copyright
The materials for this workshop are distributed under a CCY-BY-4 License and the code is made available under a GPLv3 License. Please cite appropriately the Ersilia Open Source Initiative and the H3D Foundation when using our materials.

## Disclaimer
This is a prepared workshop. Data has been curated to facilitate the student's learnings and does not represent a real scenario.
