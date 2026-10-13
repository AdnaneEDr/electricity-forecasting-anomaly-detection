import io
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from energy.anomalies import events, score_detectors
from energy.data import validate_csv
from energy.forecast import predict_all

ROOT = Path(__file__).parent
st.set_page_config(page_title="Électricité : prévisions et alertes", layout="wide")
st.title("Détection d’anomalies et prévision de séries temporelles")
st.caption("UCI · puissance moyenne en kW · alertes suspectes sans annotation réelle")
mode = st.sidebar.selectbox("Résultats entraînés", ["quick", "full"])
folder = ROOT / "artifacts" / mode
if not (folder / "models.pkl").exists():
    st.info(f"Exécutez python run.py --mode {mode} pour générer les modèles.")
    st.stop()
with (folder / "models.pkl").open("rb") as f:
    bundle = pickle.load(f)
scenario = st.sidebar.selectbox("Données", ["Originales", "Injection test", "Import CSV"])
if scenario == "Import CSV":
    st.sidebar.caption("timestamp,power_kw ; kW ; dates locales sans fuseau ; pas horaire")
    confirmed = st.sidebar.checkbox("Je confirme que power_kw est une puissance moyenne en kW")
    upload = st.sidebar.file_uploader("Importer un CSV", type="csv")
    if upload is None or not confirmed:
        st.info("Import : choisissez un fichier et confirmez l’unité.")
        st.stop()
    try:
        frame = validate_csv(io.BytesIO(upload.getvalue()))
    except (ValueError, TypeError) as exc:
        st.error(str(exc))
        st.stop()
    st.warning("Modèles entraînés sur un foyer UCI : transfert vers un autre compteur non validé.")
else:
    file = "series.csv" if scenario == "Originales" else "injected_test.csv"
    frame = pd.read_csv(folder / file, parse_dates=["timestamp"])
    st.caption("Copie avec anomalies synthétiques, graine 44." if scenario != "Originales"
               else "Données originales ; valeurs manquantes complétées avec le passé.")
y = frame.power_kw.to_numpy()
times = pd.DatetimeIndex(frame.timestamp)
tabs = st.tabs(["Exploration", "Prévisions", "Alertes et rejeu", "Évaluation"])
with tabs[0]:
    dates = st.date_input("Période", value=(times[0].date(), times[-1].date()),
                          min_value=times[0].date(), max_value=times[-1].date())
    view = frame
    if len(dates) == 2:
        view = frame.loc[frame.timestamp.between(pd.Timestamp(dates[0]),
                                                pd.Timestamp(dates[1]) + pd.Timedelta(hours=23))]
    st.plotly_chart(px.line(view, x="timestamp", y="power_kw"), width="stretch")
    seasonal = frame.assign(hour=times.hour).groupby("hour").power_kw.mean().reset_index()
    st.plotly_chart(px.line(seasonal, x="hour", y="power_kw", title="Profil horaire moyen"), width="stretch")
    st.plotly_chart(px.histogram(frame, x="power_kw", nbins=60), width="stretch")
    st.write("Autocorrélations", {f"retard {lag}h": frame.power_kw.autocorr(lag)
                                 for lag in [1, 24, 168] if len(frame) > lag})
    if scenario == "Originales":
        st.caption("STL rétrospective sur train : utilise le futur, exclue du moteur d’alertes.")
        stl = pd.read_csv(folder / "stl_train.csv", parse_dates=["timestamp"])
        st.plotly_chart(px.line(stl, x="timestamp", y=["trend", "seasonal"]), width="stretch")
with tabs[1]:
    st.caption("Prévisions à 24h depuis la dernière observation ; valeurs futures inconnues.")
    if len(y) >= 168:
        extended_times = times.append(pd.date_range(times[-1] + pd.Timedelta(hours=1), periods=24, freq="h"))
        predicted = predict_all(y, extended_times, len(y), bundle["ridge"], bundle["ets_days"])
        future = pd.DataFrame({"timestamp": extended_times[-24:], **predicted})
        st.plotly_chart(px.line(future, x="timestamp", y=list(predicted)), width="stretch")
        st.download_button("Exporter les prévisions CSV", future.to_csv(index=False), "previsions.csv", "text/csv")
    else:
        st.warning("168 heures minimum pour la prévision supervisée.")
    st.dataframe(pd.read_csv(folder / "forecast_metrics.csv"))
    st.plotly_chart(px.line(pd.read_csv(folder / "horizon_metrics.csv"),
                           x="horizon", y="MAE", color="model"), width="stretch")
    historical = pd.read_csv(folder / "predictions.csv", parse_dates=["timestamp", "origin"])
    origin = st.selectbox("Origine historique du test UCI", historical.origin.unique())
    sample = historical.loc[historical.origin == origin]
    st.plotly_chart(px.line(sample, x="timestamp", y="prediction", color="model"), width="stretch")
    st.plotly_chart(px.line(sample.drop_duplicates("timestamp"), x="timestamp", y="actual",
                           title="Observations correspondantes"), width="stretch")
    residuals = sample.assign(error_kw=sample.actual - sample.prediction)
    st.plotly_chart(px.line(residuals, x="timestamp", y="error_kw", color="model",
                           title="Erreurs historiques : observation − prévision (kW)"), width="stretch")
with tabs[2]:
    st.caption("Rejeu historique causal : seules les observations jusqu’au curseur sont transmises au détecteur.")
    cutoff = st.slider("Nombre d’observations déjà reçues", 49, len(y), min(len(y), 720))
    method = st.selectbox("Détecteur", ["Robuste", "Autoencodeur"])
    received = y[:cutoff]
    scores = score_detectors(received, times[:cutoff], bundle["detectors"])[method]
    threshold = bundle["detectors"]["thresholds"][method]
    alerts = pd.DataFrame({"timestamp": times[:cutoff], "power_kw": received,
                           "score": scores, "threshold": threshold, "alert": scores > threshold})
    fig = px.line(alerts.tail(720), x="timestamp", y="power_kw")
    flagged = alerts.tail(720).loc[lambda d: d.alert]
    fig.add_scatter(x=flagged.timestamp, y=flagged.power_kw, mode="markers", name="Suspect")
    st.plotly_chart(fig, width="stretch")
    st.plotly_chart(px.line(alerts.tail(720), x="timestamp", y=["score", "threshold"]), width="stretch")
    st.dataframe(events(alerts.alert, alerts.timestamp.to_numpy()))
    st.download_button("Exporter scores et alertes CSV", alerts.to_csv(index=False), "alertes.csv", "text/csv")
with tabs[3]:
    st.write("Résultats sur anomalies synthétiques injectées exclusivement")
    st.dataframe(pd.read_csv(folder / "synthetic_metrics.csv"))
    if (folder / "injection_cases.csv").exists():
        st.write("Détection par type d’injection et délai en heures")
        st.dataframe(pd.read_csv(folder / "injection_cases.csv"))
    st.caption("Seuils : quantile 99 % de validation originale. Aucun ajustement avec les étiquettes test.")
    st.dataframe(pd.read_csv(folder / "period_metrics.csv"))
    st.write("Limites : foyer unique, fuseau inconnu, dérive, faux positifs, absence de labels réels. Aucun intervalle de prévision fourni.")
