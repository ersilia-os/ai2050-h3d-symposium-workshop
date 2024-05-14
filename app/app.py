import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

import stylia

from ersilia_client import ErsiliaClient


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
data_dir = os.path.abspath(os.path.join(root, "..", "data"))

from utils import filter_valid_smiles, draw_molecule, query_nvidia_generative_chemistry
from utils import binarize_acinetobacter_data, train_acinetobacter_ml_model, train_final_acinetobacter_ml_model, predict_acinetobacter_ml_model

st.set_page_config(layout="wide", page_title='H3D Symposium AI Workshop', page_icon=':microbe:', initial_sidebar_state='collapsed')

# About page

st.sidebar.title("About")
st.sidebar.write("This app is part of the [H3D Symposium (Livingstone, Zambia, 2024)](https://h3dfoundation.org/5th-h3d-symposium/).")
st.sidebar.write("The workshop been jointly developed by the [Ersilia Open Source Initiative](https://ersilia.io) and the [H3D Foundation](https://h3dfoundation.org/).")
st.sidebar.write("For more information about this workshop, please see code and data in this [GitHub repository](https://github.com/ersilia-os/ai-intro-workshop).")
st.sidebar.write("If you have a more advanced dataset in mind or a use case for your research, please contact us at: [hello@ersilia.io](mailto:hello@ersilia.io).")

# Main page

st.title(":microbe: H3D Symposium - AI for Antimicrobial Drug Discovery Workshop :pill:")

model_urls = {
    "eos9ei3": "https://eos9ei3-jvhi9.ondigitalocean.app/",
    "eos43at": "https://eos43at-boaoi.ondigitalocean.app/",
}

library_filenames = {
    "Compound library 1": "abaumannii/abaumannii_subset_0.csv",
    "Compound library 2": "abaumannii/abaumannii_subset_1.csv",
    "Compound library 3": "abaumannii/abaumannii_subset_2.csv",
    "Compound library 4": "abaumannii/abaumannii_subset_3.csv",
}

@st.cache_resource
def get_client(model_id):
    return ErsiliaClient(model_urls[model_id])

@st.cache_data
def read_library(library_filename):
    return list(pd.read_csv(os.path.join(data_dir, library_filename))["smiles"])

clients = {model_id: get_client(model_id) for model_id in model_urls.keys()}

model_titles = {model_id: client.info["card"]["Title"] for model_id, client in clients.items()}

DEFAULT_ACINETOBACTER_QUESTION = "What diseases are caused by Acinetobacter baumannii?"
DEFAULT_ACINETOBACTER_ANSWER = "Hello! Welcome to the H3D Symposium in Livingstone, Zambia. Acinetobacter baumannii can cause a variety of infections, including pneumonia, bloodstream infections, urinary tract infections, and wound infections. It is known for its ability to develop resistance to multiple antibiotics, posing a challenge for treatment."

initial_question_input = st.text_input("Ask me something about Acinetobacter baumannii", value=DEFAULT_ACINETOBACTER_QUESTION)

if initial_question_input == DEFAULT_ACINETOBACTER_QUESTION:
    st.write(DEFAULT_ACINETOBACTER_ANSWER)
else:
    pass

st.divider()

st.header("Build a machine learning model")

@st.cache_data
def load_acinetobacter_training_data():
    df = pd.read_csv(os.path.join(data_dir, "training_sets", "eos3804.csv"))
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)
    df["Mean"] = pd.to_numeric(df["Mean"])
    return df

# display metrics and slider for activity cut-off
dt = load_acinetobacter_training_data()
cols = st.columns(5)
cols[0].metric("Mean growth", round(dt["Mean"].mean(), 3))
cols[1].metric("Standard deviation", round(dt["Mean"].std(), 3))
activity_cutoff = cols[2].slider("Activity cutoff", min_value=0.1, max_value=2., value=1., step=0.001, format="%.3f")
dt = binarize_acinetobacter_data(dt, cutoff=activity_cutoff)
cols[3].metric("Number of actives", sum(dt["Binary"]))
cols[4].metric("Number of inactives", dt.shape[0] - sum(dt["Binary"]))

# display data and graph according to activity cut-off
cols = st.columns([2,1])
dt_ = dt[["SMILES", "Name", "Mean", "Binary"]]
dt_.rename(columns={"Mean": "Mean growth (OD)"}, inplace=True)
cols[0].write(dt_)
fig, ax = plt.subplots()
inactive = dt[dt["Binary"]==0]
active = dt[dt["Binary"]==1]
ax.scatter(inactive.index, inactive['Mean'], marker='o', color='blue', alpha=0.5, label='Inactive')
ax.scatter(active.index, active['Mean'], marker='o', color='red', alpha=0.5, label='Active')
ax.set_title('Growth Values', size=14)
ax.set_xlabel('Molecule Index', size=12)
ax.set_ylabel('Mean Growth', size=12)
ax.legend(fontsize=12)
#cols[1].pyplot(fig)
dt_["Molecule index"] = dt_.index
dt_["color"]=['Active' if x==1 else 'Inactive' for x in dt["Binary"]]
dt_["Active"] = [dt_["Mean growth (OD)"].iloc[i] if dt_["Binary"].iloc[i] == 1 else None for i in range(len(dt_)) ]
dt_["Inactive"] = [dt_["Mean growth (OD)"].iloc[i] if dt_["Binary"].iloc[i] == 0 else None for i in range(len(dt_)) ]
cols[1].write("Mean Growth (OD) of A.baumannii")
cols[1].scatter_chart(dt_, x="Molecule index", y=["Active", "Inactive"], color=['#FF0000', '#0000FF'], size=50)

cols = st.columns(4)
if 'train_ml_model_active' not in st.session_state:
    st.session_state['train_ml_model_active'] = False

def toggle_train_ml_model_state():
    st.session_state['train_ml_model_active'] = not st.session_state['train_ml_model_active']

if cols[0].button('🤖 Train a machine learning model!', on_click=toggle_train_ml_model_state):
    if not st.session_state["train_ml_model_active"]:
        pass
    else:
        with st.spinner("Training the model..."):
            st.session_state.model_results = train_acinetobacter_ml_model(dt)
            cols[0].success('Model trained!')

if st.session_state["train_ml_model_active"]:
    if "model_results" in st.session_state:
        aurocs = st.session_state.model_results["aurocs"]
        std_auroc = np.std(aurocs)
        cols[0].metric("AUROC ± Std", f"{np.mean(aurocs):.3f} ± {std_auroc:.3f}")
        tprs = []
        mean_fpr = np.linspace(0,1,100)
        for i in st.session_state.model_results["cv_data"]:
            fpr, tpr, _ = metrics.roc_curve(i[0],i[1])
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
    tprs_df = pd.DataFrame({
        'tpr_cv1': tprs[0],
        'tpr_cv2': tprs[1],
        'tpr_cv3': tprs[2],
        'tpr_cv4': tprs[3],
        'tpr_cv5': tprs[4],
        'Mean TPR': mean_tpr,
        'FPR': mean_fpr,
        
    })

    cols[1].write("ROC Curve")
    cols[1].line_chart(data=tprs_df, x="FPR", y=["tpr_cv1","tpr_cv2","tpr_cv3","tpr_cv4","tpr_cv5","Mean TPR"], color=["#1D6996","#d3d3d3","#d3d3d3","#d3d3d3","#d3d3d3","#d3d3d3"], width=0, height=0, use_container_width=True)
    X = st.session_state.model_results["X"]
    y = st.session_state.model_results["y"]
    X_ = [x[:2] for x in X]
    LolP1 = [arr[0] for arr in X]
    LolP2 = [arr[1] for arr in X]
    lolp_df = pd.DataFrame({
        'LolP1': LolP1,
        'LolP2': LolP2,
        'Binary': y,
        'Color': ['#0000FF' if x==0 else  '#FF0000'for x in y]
    })  
    cols[2].write("2D representation of chemical space")
    cols[2].scatter_chart(data=lolp_df, x="LolP1", y="LolP2", color="Color", size=50)

    # Initialize the session state if not already done
    if 'final_model' not in st.session_state:
        st.session_state['final_model'] = False

    def toggle_final_model():
        # Toggle the state
        st.session_state['final_model'] = not st.session_state['final_model']
# Button to toggle the state
    if st.button('💾 Best cut-off selected & model trained!', on_click=toggle_final_model):
        pass

    if st.session_state["final_model"]:
        st.divider()
        st.header("Library selection")

        cols = st.columns(5)
        selected_library = cols[0].radio("Select a screening library", list(library_filenames.keys()))
        smiles_list = read_library(library_filenames[selected_library])

        smiles_list = st.text_area("Enter molecules as SMILES strings", key="smiles", value=os.linesep.join(smiles_list)).split(os.linesep)
        smiles_list = filter_valid_smiles(smiles_list)

        cols[1].metric("Number of molecules", len(smiles_list))

        library_molecules_list = [(i, smiles) for i, smiles in enumerate(smiles_list)]

        # Get the total number of molecules
        num_molecules = len(library_molecules_list)

        # Calculate the number of chunks
        num_chunks_of_3 = (num_molecules + 2) // 3

        # Initialize the current chunk index
        chunk_of_3_index = st.session_state.get('chunk_of_3_index', 0)

        # Get the current chunk of molecules
        start_index = chunk_of_3_index * 3
        end_index = min(start_index + 3, num_molecules)
        current_chunk_of_3 = library_molecules_list[start_index:end_index]

        def draw_molecules_in_chunk_of_3(cols, current_chunk):
            i = 0
            for m in current_chunk:
                cols[i+2].image(draw_molecule(m[1]), caption=f"Molecule {m[0]}")
                i += 1

        draw_molecules_in_chunk_of_3(cols, current_chunk_of_3)

        # Add a more button
        if cols[1].button("View more molecules"):
            chunk_of_3_index = (chunk_of_3_index + 1) % num_chunks_of_3

        # Update the session state
        st.session_state['chunk_of_3_index'] = chunk_of_3_index


        # Initialize the session state if not already done
        if 'hit_prioritization_active' not in st.session_state:
            st.session_state['hit_prioritization_active'] = False

        def toggle_hit_prioritization_state():
            # Toggle the state
            st.session_state['hit_prioritization_active'] = not st.session_state['hit_prioritization_active']

        # Button to toggle the state
        if st.button('♻️ Proceed to hit prioritization / Refresh', on_click=toggle_hit_prioritization_state):
            pass


        if st.session_state["hit_prioritization_active"]:

            st.divider()

            st.header("Hit prioritization")
            cont = st.container(border=None)
            cont.write("In this workshop we have pre-selected the models we will use for prioritization. Feel free to browse the Ersilia Model Hub to discover more AI/ML tools!")
            cols = st.columns(3)
            cols[0].markdown(f"""
            <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                <h5>🦠 A. baumannii Bioactivity</h5>
                <p>This is the model we just trained on a dataset of ~7500 molecules described in Liu et al, 2023 to elucidate whether a molecule is active against A.baumannii.</p>
            </div>
            """, unsafe_allow_html=True)
            cols[1].markdown(f"""
            <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                <h5>🫀 hERG Inhibition</h5>
                <p>This model is described in Jiménez-Luna et al, 2021. It was trained on a publicly available dataset with the goal of predicting hERG-mediated cardiotoxicity.</p>
            </div>
            """, unsafe_allow_html=True)
            cols[2].markdown(f"""
            <div style="background-color: lightblue; padding: 10px; border-radius: 5px;">
                <h5>🧪 Synthetic Accessibility</h5>
                <p>The synthetic accessibility score was developed by Ertl & Schuffenhauer, 2009. It estimates if a molecule will be accessible for synthesis in the laboratory.</p>
            </div>
            """, unsafe_allow_html=True)
            st.write("")

            @st.cache_data
            def run_predictive_models(model_ids, smiles_list):
                results = {}
                for  model_id in model_ids:
                    client = clients[model_id]
                    result = client.run(smiles_list)
                    results[model_id] = result
                df = pd.DataFrame({"SMILES": smiles_list})
                for model_id, result in results.items():
                    columns = list(result.columns)
                    columns = [column for column in columns if column != "input"]
                    df = pd.concat([df, result[columns]], axis=1)
                return df

            dp = None

            # Initialize the session state if not already done
            if 'model_predictions_active' not in st.session_state:
                st.session_state['model_predictions_active'] = False

            def toggle_model_predictions_state():
                # Toggle the state
                st.session_state['model_predictions_active'] = not st.session_state['model_predictions_active']

            # Button to toggle the state
            if st.button(":rocket: Run predictions!", on_click=toggle_model_predictions_state):
                pass

            if st.session_state["model_predictions_active"]: 
                with st.spinner("Running models..."):
                    dp = run_predictive_models(["eos9ei3", "eos43at"], smiles_list)
                    output_cols = {"outcome":"SA Score", "pic50": "hERG"}
                    dp.rename(columns=output_cols, inplace=True)
                    red = st.session_state.model_results["reducer"]
                    mdl = st.session_state.model_results["model"]
                    abau_preds = predict_acinetobacter_ml_model(smiles_list, red, mdl)
                    dp["Abaumannii"] = abau_preds
        

            start_hit_expansion = False
            if dp is not None:
                
                st.dataframe(dp)

                # Initialize the session state if not already done
                if 'hit_expansion_active' not in st.session_state:
                    st.session_state['hit_expansion_active'] = False

                def toggle_hit_expansion_state():
                    # Toggle the state
                    st.session_state['hit_expansion_active'] = not st.session_state['hit_expansion_active']

                # Button to toggle the state
                if st.button("♻️ Proceed to hit expansion / Refresh", on_click=toggle_hit_expansion_state):
                    pass

                if st.session_state["hit_expansion_active"]: 

                    st.header("Hit expansion")

                    seed_smiles = st.text_input("Enter a seed molecule SMILES string", value=dp.head(1)["SMILES"].values[0])

                    cols = st.columns(3)
                    cols[0].text("Seed molecule")
                    cols[0].image(draw_molecule(seed_smiles))

                    opt_property = cols[1].radio("Property to optimize", ["QED", "logP"], index=0)
                    minimize = cols[1].checkbox("Minimize property", value=False)
                    num_molecules = cols[2].slider("Number of molecules", min_value=1, max_value=100, value=30)
                    minimum_similarity = cols[2].slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.85)

                    def run_generative_models(seed_smiles, opt_property, num_molecules, minimize, minimum_similarity):
                        #return sorted([(random.choice(smiles_list), random.randint(0,100)/100) for _ in range(num_molecules)], key=lambda x: x[1], reverse=minimize)
                        return query_nvidia_generative_chemistry(smiles=seed_smiles, property=opt_property, minimize=minimize, num_molecules=num_molecules, minimum_similarity=minimum_similarity)

                    # Initialize the session state if not already done
                    if 'generate_molecules_active' not in st.session_state:
                        st.session_state['generate_molecules_active'] = False

                    def toggle_generate_molecules_state():
                        # Toggle the state
                        st.session_state['generate_molecules_active'] = not st.session_state['generate_molecules_active']

                    # Button to toggle the state
                    if st.button("🔮 Generate molecules!", on_click=toggle_generate_molecules_state):
                        pass

                    dg = None

                    if st.session_state["generate_molecules_active"]:
                        dg = run_generative_models(seed_smiles=seed_smiles, opt_property=opt_property, num_molecules=num_molecules, minimize=minimize, minimum_similarity=minimum_similarity)
                        
                    if dg is not None:
                        dg = pd.DataFrame(dg, columns=["SMILES", opt_property])
                        st.write("Virtual generated molecules")
                        cols = st.columns([2, 1, 1, 1])
                        cols[0].dataframe(dg)
                        all_molecules = []
                        for i, v in enumerate(dg.values):
                            all_molecules += [(i, v[0], v[1])]
                        # Get the total number of molecules
                        num_molecules = len(all_molecules)
                        # Calculate the number of chunks
                        num_chunks_of_6 = (num_molecules + 5) // 6
                        # Initialize the current chunk index
                        chunk_of_6_index = st.session_state.get('chunk_of_6_index', 0)
                        # Get the current chunk of molecules
                        start_index = chunk_of_6_index * 6
                        end_index = min(start_index + 6, num_molecules)
                        current_chunk_of_6 = all_molecules[start_index:end_index]
                        def draw_molecules_in_chunk_of_6(cols, current_chunk):
                            i = 0
                            for m in current_chunk:
                                if i == 3:
                                    i = 0
                                cols[i+1].image(draw_molecule(m[1]), caption=f"{m[0]}: {round(m[2], 3)}")
                                i += 1
                        draw_molecules_in_chunk_of_6(cols, current_chunk_of_6)
                        # Add a more button
                        if cols[0].button(key="More chunks of 6", label="View more molecules"):
                            chunk_of_6_index = (chunk_of_6_index + 1) % num_chunks_of_6
                        # Update the session state
                        st.session_state['chunk_of_6_index'] = chunk_of_6_index

                        st.header("Synthesis planning")
                        sel_smiles = st.text_input("Enter a SMILES string", value=seed_smiles)


