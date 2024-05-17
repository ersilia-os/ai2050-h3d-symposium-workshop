import pandas as pd
import umap
import altair as alt
import numpy as np

def plot_act_inact(df): 
    #needs a datagrame with Molecule index, Mean value and Binary
    color_mapping = alt.Color('Binary:N', scale=alt.Scale(domain=[1, 0], range=['#FF0000', '#0000FF']), legend=alt.Legend(title='Molecule Activity', labelExpr="if(datum.label == '1', 'Active', 'Inactive')"))
    scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('Molecule index:Q', title='Molecule Index'),
        y=alt.Y('Mean:Q', title='Mean growth (OD)'),
        color=color_mapping
    ).properties(
        title='Molecule activity for A.baumannii'
    ).configure_title(
            anchor='middle'
        ).interactive()
    return scatter_plot

def plot_roc_curve(df):
    fig = alt.Chart(df).transform_fold(
        ['tpr_cv1', 'tpr_cv2', 'tpr_cv3', 'tpr_cv4', 'tpr_cv5', 'Mean TPR'],
        as_=['Variable', 'Value']
    ).mark_line().encode(
        x=alt.X('FPR:Q', title='False Positive Rate (FPR)'),
        y=alt.Y('Value:Q', title='True Positive Rate (TPR)'),
        color=alt.Color('Variable:N', scale=alt.Scale(range= ['#0000FF']+['#d3d3d3']*5), legend=None),
    ).properties(
        title='ROC Curve'
    ).configure_title(
        anchor='middle'
    ).interactive()
    return fig

def plot_lolp(X,y):
    X_ = [x[:2] for x in X]
    LolP1 = [arr[0] for arr in X_]
    LolP2 = [arr[1] for arr in X_]
    lolp_df = pd.DataFrame({
        'LolP1': LolP1,
        'LolP2': LolP2,
        'Binary': y,
        'Color': ['#0000FF' if x==0 else  '#FF0000'for x in y]
    })  
    lolp_df_sorted = lolp_df.sort_values(by='Binary', ascending=True)
    fig = alt.Chart(lolp_df_sorted).mark_circle(size=60).encode(
    x=alt.X('LolP1', title='LolP1'),
    y=alt.Y('LolP2', title='LolP2'),
    color=alt.Color('Binary:N', scale=alt.Scale(domain=[0, 1], range=['#0000FF', '#FF0000']), legend=None)
    ).properties(
        title='2D Representation of Chemical Space'
    ).configure_title(
        anchor='middle'
    ).interactive()
    return fig


def plot_umap(X, y):
    y = np.array(y)
    reducer = umap.UMAP(n_components=2)
    X_embedded = reducer.fit_transform(X, y)
    df = pd.DataFrame({
        'UMAP1': X_embedded[:, 0],
        'UMAP2': X_embedded[:, 1],
        'Binary': y,
        'Color': ['#0000FF' if x == 0 else '#FF0000' for x in y]
    })
    # Sort the DataFrame so that inactive molecules (blue) are plotted first
    df_sorted = df.sort_values(by='Binary', ascending=True)
    # Create the Altair scatter plot
    fig = alt.Chart(df_sorted).mark_circle(size=60).encode(
        x=alt.X('UMAP1', title='UMAP 1'),
        y=alt.Y('UMAP2', title='UMAP 2'),
        color=alt.Color('Binary:N', scale=alt.Scale(domain=[0, 1], range=['#0000FF', '#FF0000']), legend=None),
        tooltip=['UMAP1', 'UMAP2', 'Binary']
    ).properties(
        title='2D UMAP Projection of Molecules'
    ).configure_title(
        anchor='middle'
    ).interactive()
    return fig