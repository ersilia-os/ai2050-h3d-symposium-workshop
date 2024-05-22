import os
import sys
import random
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn import metrics

from ersilia_client import ErsiliaClient

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
data_dir = os.path.abspath(os.path.join(root, "..", "data"))

from utils import filter_valid_smiles, draw_molecule
from utils import query_nvidia_generative_chemistry, ask_question_about_abaumannii
from utils import binarize_acinetobacter_data, train_acinetobacter_ml_model, predict_acinetobacter_ml_model
from info import about, model_urls_do, model_urls_aws, library_filenames, q1, q2, q3
from info import abaumannii_bioactivity, herg_inhibition, synthetic_accessibility
from plots import plot_act_inact, plot_roc_curve, plot_lolp, plot_umap
from chemspace import ChemSpaceSearch

model_urls=model_urls_do

st.set_page_config(layout="wide", page_title='H3D Symposium AI Workshop', page_icon=':microbe:', initial_sidebar_state='collapsed')

@st.cache_resource
def get_client(model_id):
    return ErsiliaClient(model_urls[model_id])

@st.cache_data
def read_library(library_filename):
    return list(pd.read_csv(os.path.join(data_dir, library_filename))["smiles"])

clients = {model_id: get_client(model_id) for model_id in model_urls.keys()}

# ABOUT
st.sidebar.title("About")
for i in range(4):
    st.sidebar.write(about[i])
    
# MAIN
st.title(":microbe: H3D Symposium - AI for Antimicrobial Drug Discovery Workshop :pill:")

# Section 1: Open AI question

DEFAULT_ACINETOBACTER_QUESTION = "What diseases are caused by Acinetobacter baumannii?"
DEFAULT_ACINETOBACTER_ANSWER = "Hello! Welcome to the H3D Symposium in Livingstone, Zambia. Acinetobacter baumannii can cause a variety of infections, including pneumonia, bloodstream infections, urinary tract infections, and wound infections. It is known for its ability to develop resistance to multiple antibiotics, posing a challenge for treatment."

initial_question_input = st.text_input("Ask me something about Acinetobacter baumannii", value=DEFAULT_ACINETOBACTER_QUESTION)

@st.cache_data(show_spinner=False)
def ask_initial_question(query):
    return ask_question_about_abaumannii(query)

if initial_question_input == DEFAULT_ACINETOBACTER_QUESTION:
    st.write(DEFAULT_ACINETOBACTER_ANSWER)
else:
    text = ask_initial_question(initial_question_input)
    st.write(text)


# Section 2: Build a Model
st.divider()
st.header("Build a machine learning model")

@st.cache_data(show_spinner=False)
def load_acinetobacter_training_data():
    df = pd.read_csv(os.path.join(data_dir, "training_sets", "eos3804.csv"))
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    df.reset_index(drop=True, inplace=True)
    df["Mean"] = pd.to_numeric(df["Mean"])
    return df

@st.cache_data(show_spinner=False)
def do_plot_roc_curve(tprs_df):
    return plot_roc_curve(tprs_df)

@st.cache_data(show_spinner=False)
def do_plot_lolp(X, y):
    return plot_lolp(X, y)

@st.cache_data(show_spinner=False)
def do_plot_umap(X, y):
    return plot_umap(X, y)

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
cols[0].write(dt_)
dt_["Molecule index"] = dt_.index
fig = plot_act_inact(dt_)
cols[1].altair_chart(fig, use_container_width=True)

# Button to train a model & see results
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
if st.session_state["train_ml_model_active"]:
    if "model_results" in st.session_state:
        aurocs = st.session_state.model_results["aurocs"]
        std_auroc = np.std(aurocs)
        
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
    X = st.session_state.model_results["X"]
    y = st.session_state.model_results["y"]
    fig1 = do_plot_roc_curve(tprs_df)
    fig2 = do_plot_lolp(X, y)
    fig3 = do_plot_umap(X, y)
    cols[0].metric("AUROC ± Std", f"{np.mean(aurocs):.3f} ± {std_auroc:.3f}")
    cols[0].success('Model trained!')
    cols[1].altair_chart(fig1, use_container_width=True)
    cols[2].altair_chart(fig2, use_container_width=True)
    cols[3].altair_chart(fig3,use_container_width=True)
            

    q1_header = "Ask yourselves the following questions:"
    st.write(q1_header)
    q_comb = '  \n'.join(q1)
    st.info(q_comb)
    q1_closing = "Once you have decided the best cut-off and trained the model, click below to proceed:"
    st.write(q1_closing)

# button to save final model with right cut-off and proceed
    if 'final_model' not in st.session_state:
        st.session_state['final_model'] = False
    def toggle_final_model():
        st.session_state['final_model'] = not st.session_state['final_model']
    if st.button('💾 Best cut-off selected & model trained!', on_click=toggle_final_model):
        pass

# Section 3: Select Library for prediction

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

# Section 4: Hit Prioritization
        if 'hit_prioritization_active' not in st.session_state:
            st.session_state['hit_prioritization_active'] = False
        def toggle_hit_prioritization_state():
            st.session_state['hit_prioritization_active'] = not st.session_state['hit_prioritization_active']
        if st.button('♻️ Proceed to hit prioritization / Refresh', on_click=toggle_hit_prioritization_state):
            if not st.session_state["hit_prioritization_active"]:
                pass

        if st.session_state["hit_prioritization_active"]:

            st.divider()
            st.header("Hit prioritization")

            st.write("In this workshop we have pre-selected the models we will use for prioritization. Feel free to browse the Ersilia Model Hub to discover more AI/ML tools!")
            cols = st.columns(3)
            cols[0].markdown(abaumannii_bioactivity, unsafe_allow_html=True)
            cols[1].markdown(herg_inhibition, unsafe_allow_html=True)
            cols[2].markdown(synthetic_accessibility, unsafe_allow_html=True)
            st.write("")

            st.toast("Running Ersilia models")
            @st.cache_data(show_spinner=False)
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
            q2_header = "Before continuing, think about..."
            st.write(q2_header)
            q_comb = '  \n'.join(q2)
            st.info(q_comb)

# button to run predictions
            if 'model_predictions_active' not in st.session_state:
                st.session_state.dp = None
                st.session_state['model_predictions_active'] = False
            def toggle_model_predictions_state():
                st.session_state['model_predictions_active'] = not st.session_state['model_predictions_active']
            if st.button(":rocket: Run predictions!", on_click=toggle_model_predictions_state):
                if not st.session_state["model_predictions_active"]:
                    pass
                else:
                    with st.spinner("Running models..."):
                        dp = run_predictive_models(["eos9ei3", "eos43at"], smiles_list)
                        output_cols = {"sa_score":"SA Score", "pic50": "hERG"}
                        dp.rename(columns=output_cols, inplace=True)
                        red = st.session_state.model_results["reducer"]
                        mdl = st.session_state.model_results["model"]
                        st.toast("Running the Acinetobacter model")
                        abau_preds = predict_acinetobacter_ml_model(smiles_list, red, mdl)
                        dp["Abaumannii"] = abau_preds
                        st.session_state.dp = dp

            cols = st.columns(2)
            start_hit_expansion = False

            if st.session_state.dp is not None:
                dp =st.session_state.dp
                cols[0].dataframe(dp)
                if len(dp)<50:
                    cols[1].error("Make predictions for more than 50 molecules to obtain a plot.")
                else:
                    model_column = cols[1].selectbox("Select Model Column", ["SA Score", "hERG", "Abaumannii"])
                    # Plot the selected model column in a histogram
                    hist_data = pd.DataFrame({model_column: dp[model_column]})
                    # Plot the histogram
                    chart = alt.Chart(hist_data).mark_bar(color="#1D6996").encode(
                        alt.X(model_column, bin=alt.Bin(maxbins=30), axis=alt.Axis(title=f'{model_column} prediction')),
                        y=alt.Y('count()', axis=alt.Axis(title='Counts'))
                    ).properties(
                        title=f'Histogram of {model_column} predictions'
                    )
                    cols[1].altair_chart(chart, use_container_width=True)

# Section 5: Hit Expansion
                if 'hit_expansion_active' not in st.session_state:
                    st.session_state['hit_expansion_active'] = False
                def toggle_hit_expansion_state():
                    st.session_state['hit_expansion_active'] = not st.session_state['hit_expansion_active']
                if st.button("♻️ Proceed to hit expansion / Refresh", on_click=toggle_hit_expansion_state):
                    if not st.session_state["hit_expansion_active"]:
                        pass

                if st.session_state["hit_expansion_active"]: 
                    st.divider()
                    st.header("Hit expansion")
                    seed_smiles = st.text_input("Enter a seed molecule SMILES string", value=dp.head(1)["SMILES"].values[0])

                    cols = st.columns(3)
                    cols[0].text("Seed molecule")
                    cols[0].image(draw_molecule(seed_smiles))

                    opt_property = cols[1].radio("Property to optimize", ["QED", "logP"], index=0)
                    minimize = cols[1].checkbox("Minimize property", value=False)
                    num_molecules = cols[2].slider("Number of molecules", min_value=1, max_value=100, value=30)
                    minimum_similarity = cols[2].slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.85)
                    
                    @st.cache_data(show_spinner=False)
                    def run_generative_models(seed_smiles, opt_property, num_molecules, minimize, minimum_similarity):
                        return sorted([(random.choice(smiles_list), random.randint(0,100)/100) for _ in range(num_molecules)], key=lambda x: x[1], reverse=minimize)
                        #return query_nvidia_generative_chemistry(smiles=seed_smiles, property=opt_property, minimize=minimize, num_molecules=num_molecules, minimum_similarity=minimum_similarity)

                    # Button to generate new molecules
                    if 'generate_molecules_active' not in st.session_state:
                        st.session_state['generate_molecules_active'] = False
                    def toggle_generate_molecules_state():
                        st.session_state['generate_molecules_active'] = not st.session_state['generate_molecules_active']
                    if st.button("🔮 Generate molecules!", on_click=toggle_generate_molecules_state):
                        if not st.session_state["generate_molecules_active"]:
                            pass
                        else:
                            with st.spinner("Generating molecules..."):
                                st.session_state.dg = run_generative_models(seed_smiles=seed_smiles, opt_property=opt_property, num_molecules=num_molecules, minimize=minimize, minimum_similarity=minimum_similarity)

                    if "dg" in st.session_state:
                        dg = pd.DataFrame(st.session_state.dg, columns=["SMILES", opt_property])
                        print(len(dg))
                        dg_ = dg.drop_duplicates(subset=["SMILES"])
                        print(len(dg_))
                        st.write("Virtual generated molecules")
                        cols = st.columns([2, 1, 1, 1])
                        cols[0].dataframe(dg_)
                        all_molecules = []
                        for i, v in enumerate(dg_.values):
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
                        
                        q3_header = "Before moving on, discuss about:"
                        st.write(q3_header)
                        q_comb = '  \n'.join(q3)
                        st.info(q_comb)

                        # Section 6: synthesis planning
                        if 'synthesis_planning' not in st.session_state:
                            st.session_state['synthesis_planning'] = False
                        def toggle_hit_expansion_state():
                            st.session_state['synthesis_planning'] = not st.session_state['synthesis_planning']
                        if st.button("♻️ Proceed to molecule purchasing", on_click=toggle_hit_expansion_state):
                            if not st.session_state["synthesis_planning"]:
                                pass
                        if st.session_state["synthesis_planning"]:
                            st.divider()
                            st.header("Purchase molecule")
                            cols = st.columns([2,1])
                            sel_smiles = cols[0].text_input("Enter a SMILES string", value=seed_smiles)
                            cols[1].text("Candidate molecule")
                            cols[1].image(draw_molecule(sel_smiles))
                            cols[0].info("This tool performs an automated search of Chem-Space to identify in-stock molecules for direct purchasing")
                            chemspace = ChemSpaceSearch()
                            data = chemspace.run(sel_smiles)
                            cols[0].dataframe(data)
                            


