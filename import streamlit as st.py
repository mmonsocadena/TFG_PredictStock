import os
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# --- DIRECTORIS I MAPATGE ---
BASE_DATA_DIR    = os.path.join("Conjunt de dades Preprocessades", "Datasets")
BASE_RESULTS_DIR = "."  # els models estan al root

DATASETS = {
    'Amazon':        'Amazon_Stock_Price_output.csv',
    'Google':        'Google_Stock_Price_output.csv',
    'Euro Stoxx 50': 'Euro_Stoxx_50_Stock_Price_output.csv',
    'Hang Seng':     'Hang_Seng_Stock_Price_output.csv',
    'IBEX 35':       'IBEX_35_Stock_Price_output.csv',
    'Indra':         'Indra_Stock_Price_output.csv',
    'P&G':           'P&G_Stock_Price_output.csv',
    'S&P 500':       'S&P500_Stock_Price_output.csv',
}

MODELS = [
    'Ridge Regression',
    'Random Forest',
    'Support Vector Regression (SVR)',
    'LSTM',
    'GRU',
    'Híbrido LSTM+XGBoost'
]

MODEL_RESULT_SUBDIR = {
    'Ridge Regression':            os.path.join('RIDGE REGRESSION', 'RIDGE', 'resultats_RIDGE'),
    'Random Forest':               os.path.join('RANDOM FOREST', 'resultats_RANDOM_FOREST'),
    'Support Vector Regression (SVR)': os.path.join('SVR', 'resultats_SVR'),
    'LSTM':                        os.path.join('LSTM', 'resultats_LSTM'),
    'GRU':                         os.path.join('GRU', 'resultats_GRU_Attention'),
    'Híbrido LSTM+XGBoost':       os.path.join('LSTM', 'RESULTATS_HIBRIDS'),
}

# --- STREAMLIT UI ---
st.set_page_config(page_title="Predicció del preu de les accions a 10 dies", layout="wide")
st.title("Predicció del preu de les accions a 10 dies")

st.sidebar.header("Paràmetres")
dataset_name = st.sidebar.selectbox("Escull un dataset", list(DATASETS.keys()))
model_name   = st.sidebar.selectbox("Escull un model", MODELS)
run_btn      = st.sidebar.button("Mostra prediccions & plot")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

if run_btn:
    # 1) Carregar el dataset (sense mostrar preview)
    csv_path = os.path.join(BASE_DATA_DIR, DATASETS[dataset_name])
    if not os.path.isfile(csv_path):
        st.error(f"No trobo el fitxer de dades:\n`{csv_path}`")
        st.stop()
    df = load_data(csv_path)

    # 2) Buscar la carpeta de resultats del model
    subdir = MODEL_RESULT_SUBDIR.get(model_name)
    if not subdir:
        st.error(f"No hi ha configurat resultats per a “{model_name}”")
        st.stop()

    model_dir = os.path.join(BASE_RESULTS_DIR, subdir)
    if not os.path.isdir(model_dir):
        st.error(f"No existeix la carpeta de resultats:\n`{model_dir}`")
        st.stop()

    # 3) Localitzar la subcarpeta del dataset dins el directori del model
    ds_folder = None
    for d in os.listdir(model_dir):
        if dataset_name.replace(' ', '').lower() in d.replace('_','').lower():
            ds_folder = d
            break
    if ds_folder is None:
        st.error(f"No trobo cap subcarpeta per a “{dataset_name}” dins `{model_dir}`")
        st.stop()

    result_ds_dir = os.path.join(model_dir, ds_folder)
    files         = os.listdir(result_ds_dir)

    # 4) Trobar els fitxers de futures
    csv_fut  = next((f for f in files if f.lower().endswith('.csv')  and 'future' in f.lower()), None)
    html_fut = next((f for f in files if f.lower().endswith('.html') and 'future' in f.lower()), None)

    # 5) Mostrar el CSV de prediccions a 10 dies
    if csv_fut:
        df_fut = pd.read_csv(os.path.join(result_ds_dir, csv_fut), index_col=0, parse_dates=True)
        st.subheader("📊 Prediccions a 10 dies")
        st.dataframe(df_fut)
    else:
        st.warning("No s'ha trobat cap fitxer CSV de prediccions a 10 dies.")

    # 6) Mostrar el plot HTML interactiu
    if html_fut:
        html_path = os.path.join(result_ds_dir, html_fut)
        with open(html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
        st.subheader("📈 Visualització de la predicció")
        components.html(html_data, height=500, scrolling=True)
    else:
        st.warning("No s'ha trobat cap fitxer HTML de visualització.")
