import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import plotly.graph_objects as go

# ============================
#  Funcions auxiliars
# ============================

def recompute_indicators(df):
    """
    Recalcula els indicadors tècnics morts al teu codi original:
    EMA_7, EMA_40, MACD, Signal_Line, MACD_Hist, RSI, ATR.
    Aquestes s’escriuen directament a df.
    """
    # EMA 7 i EMA 40
    df['EMA_7']  = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_40'] = df['Close'].ewm(span=40, adjust=False).mean()

    # MACD (EMA_12 - EMA_26), Signal Line (EMA de MACD), MACD Hist
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # RSI (Relative Strength Index) de 14 períodes
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range) de 14 períodes
    high_low = df['High'] - df['Low']
    high_prev_close = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close  = (df['Low'] - df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14, min_periods=14).mean()

    # Les primeres 13-14 files quedaran NaN perquè no hi ha prou dades per als indicadors.
    return

def run_svr_pipeline(df, dataset_name="dataset"):
    """
    Rep un DataFrame amb columnes: ['Date','Open','High','Low','Close','Volume', ...]
    (i qualsevol altra columna), recalcula indicadors, fa split 70/15/15, escala,
    cerca d’hiperparàmetres, entrena SVR i obté mètriques. Retorna:
      - df_result_metrics: DataFrame amb MAE, RMSE, R2 per al test
      - fig_test: figura Plotly de real vs. predicció sobre TEST
      - df_future_preds: DataFrame amb dates futures + prediccions
      - fig_future: figura Plotly de històric + prediccions futures
    """
    df_proc = df.copy().reset_index(drop=True)

    # 1) Recalcular indicadors
    recompute_indicators(df_proc)

    # 2) Definir features i target
    FEATURE_COLUMNS = [
        'Open', 'High', 'Low', 'Volume',
        'EMA_7', 'EMA_40', 'MACD', 'Signal_Line',
        'MACD_Hist', 'RSI', 'ATR'
    ]
    TARGET_COLUMN = 'Close'

    # 3) Dropna en les columnes rellevants
    before = len(df_proc)
    df_proc.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN], inplace=True)
    after = len(df_proc)
    dropped = before - after

    # 4) Split cronològic 70% train / 15% val / 15% test
    n_total = len(df_proc)
    train_size = int(n_total * 0.70)
    val_size   = int(n_total * 0.15)
    test_size  = n_total - train_size - val_size

    X_raw = df_proc[FEATURE_COLUMNS].values
    y_raw = df_proc[TARGET_COLUMN].values

    X_train_raw = X_raw[:train_size]
    y_train_raw = y_raw[:train_size]

    X_val_raw   = X_raw[train_size : train_size + val_size]
    y_val_raw   = y_raw[train_size : train_size + val_size]

    X_test_raw  = X_raw[train_size + val_size :]
    y_test_raw  = y_raw[train_size + val_size :]

    dates = df_proc['Date']
    dates_test = dates.iloc[train_size + val_size :].reset_index(drop=True)

    # 5) Escalat (StandardScaler)
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val   = scaler_X.transform(X_val_raw)
    X_test  = scaler_X.transform(X_test_raw)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
    y_val   = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
    y_test  = scaler_y.transform(y_test_raw.reshape(-1, 1)).ravel()

    # 6) Cerca d’hiperparàmetres SVR (TimeSeriesSplit CV)
    svr = SVR(kernel='rbf')
    param_grid = {
        'C':       [1, 10, 100, 1000],
        'gamma':   [0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 1]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    # 7) Entrenament final amb els millors paràmetres
    best_svr = SVR(
        kernel='rbf',
        C=best_params['C'],
        gamma=best_params['gamma'],
        epsilon=best_params['epsilon']
    )
    best_svr.fit(X_train, y_train)

    # 8) Avaluació sobre VALIDATION i TEST (desescalem i calculem mètriques)
    y_val_pred_scaled  = best_svr.predict(X_val)
    y_val_pred         = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
    y_val_true         = y_val_raw

    y_test_pred_scaled = best_svr.predict(X_test)
    y_test_pred        = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    y_test_true        = y_test_raw

    mse_val = mean_squared_error(y_val_true, y_val_pred)
    mae_val = mean_absolute_error(y_val_true, y_val_pred)
    r2_val  = r2_score(y_val_true, y_val_pred)

    mse_test = mean_squared_error(y_test_true, y_test_pred)
    mae_test = mean_absolute_error(y_test_true, y_test_pred)
    r2_test  = r2_score(y_test_true, y_test_pred)

    # 9) Gràfica Real vs. Predicho al TEST (Plotly)
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(
        x=dates_test,
        y=y_test_true,
        mode='lines',
        name='Actual Close',
        line=dict(color='blue')
    ))
    fig_test.add_trace(go.Scatter(
        x=dates_test,
        y=y_test_pred,
        mode='lines',
        name='Predicted Close',
        line=dict(color='red', dash='dash')
    ))
    fig_test.update_layout(
        title=f"{dataset_name.capitalize()} – Real vs. Predicción (TEST)",
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_white',
        xaxis_rangeslider_visible=True
    )

    # 10) Autopredict els propers 10 dies laborables
    # ----------------------------------------------
    df_future = df_proc.copy().reset_index(drop=True)
    future_dates = pd.bdate_range(
        start=df_future['Date'].iloc[-1] + pd.Timedelta(days=1),
        periods=10
    )
    future_preds = []

    for date in future_dates:
        last_row = df_future.iloc[-1]
        feature_values = last_row[FEATURE_COLUMNS].values.reshape(1, -1)

        feature_scaled = scaler_X.transform(feature_values)
        y_pred_scaled  = best_svr.predict(feature_scaled)[0]
        y_pred_real    = scaler_y.inverse_transform([[y_pred_scaled]])[0][0]
        future_preds.append(y_pred_real)

        # Construir nova fila per recalcular indicadors
        prev = df_future.iloc[-1]
        new_index = len(df_future)
        df_future.loc[new_index, 'Date']        = date
        df_future.loc[new_index, 'Open']        = prev['Close']
        df_future.loc[new_index, 'High']        = y_pred_real
        df_future.loc[new_index, 'Low']         = y_pred_real
        df_future.loc[new_index, 'Close']       = y_pred_real
        df_future.loc[new_index, 'Volume']      = prev['Volume']
        df_future.loc[new_index, 'EMA_7']       = np.nan
        df_future.loc[new_index, 'EMA_40']      = np.nan
        df_future.loc[new_index, 'MACD']        = np.nan
        df_future.loc[new_index, 'Signal_Line'] = np.nan
        df_future.loc[new_index, 'MACD_Hist']   = np.nan
        df_future.loc[new_index, 'RSI']         = np.nan
        df_future.loc[new_index, 'ATR']         = np.nan

        recompute_indicators(df_future)

    # 11) Resultats finals
    df_future_preds = pd.DataFrame({
        "Date":           future_dates,
        "Predicted_Close": future_preds
    })

    # 12) Gràfica Històric + Prediccions futures (Plotly)
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(
        x=df_proc['Date'], y=df_proc['Close'],
        mode='lines', name='Histórico Close',
        line=dict(color='lightblue')
    ))
    fig_future.add_trace(go.Scatter(
        x=future_dates, y=np.array(future_preds),
        mode='lines+markers', name='Predicción futura',
        line=dict(color='orange', dash='dash'),
        marker=dict(size=6)
    ))
    fig_future.update_layout(
        title=f"{dataset_name.capitalize()} – Predicción Próximos 10 Días",
        xaxis_title='Date',
        yaxis_title='Close Price',
        template='plotly_white',
        xaxis_rangeslider_visible=True
    )

    # 13) Mètriques en DataFrame
    df_metrics = pd.DataFrame({
        "Set":     ["Validation", "Test"],
        "MSE":     [mse_val, mse_test],
        "MAE":     [mae_val, mae_test],
        "R2":      [r2_val, r2_test]
    })

    return df_metrics, fig_test, df_future_preds, fig_future, dropped, best_params

# ============================
#  Streamlit UI
# ============================
st.set_page_config(page_title="SVR Stock Prediction", layout="wide")

st.title("Predicció SVR del preu de les accions a 10 dies")

st.markdown("""
**Instruccions**:
1. Pugeu un fitxer CSV que contingui almenys les columnes:
   - `Date`
   - `Open`, `High`, `Low`, `Close`, `Volume`
2. L’aplicació entrenarà un SVR amb cerca d’hiperparàmetres, avaluarà el model i finalment mostrarà:
   - Les **mètriques** (MSE, MAE, R²) de validation i test  
   - La **gràfica** de real vs. predicció sobre TEST  
   - La **gràfica futura** (els pròxims 10 dies laborables)
   - I una taula amb les prediccions futures (Data + Preu predicció)
""")

uploaded_file = st.file_uploader(
    "1. Pugeu el fitxer CSV aquí", type="csv", accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        # Comprovació mínima: hi ha 'Date' i 'Close'?
        if 'Date' not in df_input.columns or 'Close' not in df_input.columns:
            st.error("El CSV ha de contenir, com a mínim, les columnes 'Date' i 'Close'.")
            st.stop()
        df_input['Date'] = pd.to_datetime(df_input['Date'], dayfirst=False)
        df_input.sort_values('Date', inplace=True)
    except Exception as e:
        st.error(f"S'ha produït un error llegint el CSV:\n```{e}```")
        st.stop()

    # Botó per iniciar tot el procés SVR
    if st.button("👉 Executar pipeline SVR i mostrar resultats"):
        with st.spinner("Executant pipeline SVR i generant resultats, espereu..."):
            # Cridem la funció que ho fa tot i recollim resultats
            (df_metrics,
             fig_test,
             df_future_preds,
             fig_future,
             dropped_rows,
             best_params) = run_svr_pipeline(df_input, dataset_name=os.path.splitext(uploaded_file.name)[0])

        # 1) Mostrem quantes files s'han eliminat per NaNs
        st.markdown(f"**Files eliminades per NaNs (indicadors tècnics sense dades suficients):** {dropped_rows}")

        # 2) Paràmetres SVR seleccionats
        st.subheader("Hiperparàmetres triats (GridSearchCV sobre l’entrenament)")
        st.write(best_params)

        # 3) Mètriques (validation vs. test)
        st.subheader("Mètriques del model")
        st.dataframe(df_metrics.style.format({
            "MSE": "{:.4f}",
            "MAE": "{:.4f}",
            "R2": "{:.4f}"
        }))

        # 4) Gràfica TEST: Real vs. Predicció
        st.subheader("→ Gràfica: Real vs. Predicció (TEST)")
        st.plotly_chart(fig_test, use_container_width=True)

        # 5) Taula amb prediccions futurs 10 dies
        st.subheader("Taula: Prediccions per als pròxims 10 dies laborables")
        st.dataframe(df_future_preds.style.format({"Predicted_Close": "{:.2f}"}))

        # 6) Gràfica futura
        st.subheader("→ Gràfica futura (últim dia històric + pròxims 10 dies)")
        st.plotly_chart(fig_future, use_container_width=True)

        # 7) Permetre descarregar el CSV de prediccions futures
        csv_buffer = df_future_preds.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descarregar CSV amb prediccions futures",
            data=csv_buffer,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_future_10days.csv",
            mime="text/csv"
        )

else:
    st.info("Puja un CSV per començar. Ha de contenir almenys les columnes: Date, Open, High, Low, Close, Volume.")

# streamlit run 'prova aplicacio 2.py'