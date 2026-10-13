import json
from pathlib import Path

import pandas as pd

root = Path(__file__).parent
quick = root / "artifacts/quick"
r = json.loads((quick / "run.json").read_text(encoding="utf-8"))
meta = json.loads((root / "data/metadata.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(quick / "forecast_metrics.csv")
synthetic = pd.read_csv(quick / "synthetic_metrics.csv")


def table(df):
    columns = list(df.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in df.itertuples(index=False, name=None):
        rows.append("| " + " | ".join("—" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(rows)


best = metrics.sort_values("MAE").iloc[0]
naive = metrics.loc[metrics.model == "Naïve"].iloc[0]
gain = 100 * (1 - best.MAE / naive.MAE)
hm = pd.read_csv(quick / "horizon_metrics.csv")
ridge_h = hm.loc[hm.model == "Ridge direct"]
worst = ridge_h.sort_values("MAE", ascending=False).iloc[0]
period = pd.read_csv(quick / "period_metrics.csv")
text = f"""# Rapport comparatif — exécution réelle

Environnement Windows, Python 3.13.5, CPU, dépendances exactes dans
`requirements.lock.txt`. Résultats ci-dessous issus de `artifacts/quick/run.json`
et des fichiers CSV produits par le pipeline, sans métrique simulée.

## Données et protocole

Archive UCI : {meta['raw_rows']} lignes minute, {meta['raw_missing']} mesures manquantes,
{meta['duplicates']} doublons ; {meta['hourly_missing']} heures insuffisamment renseignées
sur l’archive complète. Période : {meta['start']} à {meta['end']}.
SHA256 : `{meta['sha256']}`.

Mode rapide : {r['n']} heures de 2009, {r['train']} train, {r['validation']} validation,
{r['test']} test ; {r['imputed']} heures complétées. Prévision 24h, 28 origines glissantes
dans le test, cibles imputées exclues des métriques. Dénominateur MASE train :
{r['MASE_train_denominator']:.6f} kW. Les origines peuvent se recouvrir ; les résultats
ne constituent pas 672 erreurs indépendantes. Le test ne sert à aucun réglage.

## Prévision sur données originales

{table(metrics)}

Le meilleur MAE mesuré est celui de **{best.model}**, {best.MAE:.4f} kW,
soit {gain:.1f}% de réduction face à la naïve sur ces origines.
Cette observation sur un seul split ne prouve pas une supériorité générale.
Le résultat test ne modifie pas les hyperparamètres sélectionnés sur validation.

Sélection Ridge : {r['ridge_tuning']}. Sélection historique ETS : {r['ets_tuning']}.
Les erreurs par horizon sont dans `horizon_metrics.csv`, celles selon le week-end
dans `period_metrics.csv`. Les modèles sont évalués sur les mêmes origines.

Ridge : MAE à horizon 1h = {ridge_h.loc[ridge_h.horizon == 1, 'MAE'].iloc[0]:.4f} kW,
à 24h = {ridge_h.loc[ridge_h.horizon == 24, 'MAE'].iloc[0]:.4f} kW ;
horizon le plus difficile = {int(worst.horizon)}h ({worst.MAE:.4f} kW).
Une erreur croissante avec l’horizon n’est pas garantie : le cycle journalier peut
rendre certaines heures intermédiaires plus difficiles que 24h.

{table(period)}

Graphiques interactifs autonomes : [série](../artifacts/quick/series.html),
[erreurs par horizon](../artifacts/quick/horizons.html),
[comparaison synthétique](../artifacts/quick/synthetic.html).

## Détection sur anomalies injectées uniquement

{table(synthetic)}

Les seuils sont fixés sur validation originale (quantile 99%) : {r['thresholds']}.
Calibration synthétique : graine 43 ; test synthétique : graine 44, cinq événements.
Les deux méthodes reçoivent la même série altérée. Les positions/amplitudes/durées
complètes sont dans `run.json`. Aucun ajustement des prédictions avec les labels test.
AP = average precision ; le délai moyen ne concerne que les événements détectés.
Le rappel événement doit être lu avec le F1 horaire : une fenêtre d’autoencodeur
peut prolonger les alertes après la fin de l’injection et produire des faux positifs.

Les alertes sur observations originales ne sont jamais évaluées comme des anomalies
réelles confirmées. Les CSV d’événements permettent une inspection humaine.

## Limites et vérifications

Foyer unique ancien, fuseau horaire non documenté, seuil sensible à la dérive et aux
valeurs imputées, train supposé majoritairement normal. Cinq événements d’une seule
graine test : absence d’intervalle de confiance et d’évaluation métier réelle.
La dérive graduelle et les plateaux peuvent être difficiles à détecter. Pas de bande
de prévision arbitraire : aucun intervalle calibré n’est fourni.

Les détails des vérifications exécutées figurent dans `VERIFICATIONS.md`.
Le mode complet est fourni séparément et ses résultats, lorsqu’exécutés, restent
dans `artifacts/full` ; ils ne sont pas mélangés au tableau rapide ci-dessus.

## Trois puces CV fondées sur cette exécution

- Développement d’un pipeline reproductible de consommation électrique UCI :
  acquisition avec SHA256, agrégation horaire et séparation temporelle sans fuite.
- Comparaison de quatre prévisionnistes à 24h et entraînement d’un autoencodeur
  PyTorch CPU ; meilleur MAE mesuré de {best.MAE:.3f} kW sur le test 2009 retenu.
- Réalisation d’une application Streamlit en français avec rejeu causal, imports et
  exports CSV, évaluation d’anomalies injectées et tests de non-utilisation du futur.
"""
full = root / "artifacts/full"
if (full / "run.json").exists():
    full_run = json.loads((full / "run.json").read_text(encoding="utf-8"))
    full_metrics = pd.read_csv(full / "forecast_metrics.csv")
    full_synth = pd.read_csv(full / "synthetic_metrics.csv")
    text += f"""

## Mode complet : résultats distincts

Exécution effectuée sur {full_run['n']} heures : {full_run['train']} train,
{full_run['validation']} validation, {full_run['test']} test. 40 époques,
40 origines de validation et 100 origines test ; {full_run['imputed']} heures imputées.
La première heure incomplète est exclue. Les partitions ne sont pas celles de 2009 :
on ne peut attribuer une différence de performance uniquement au nombre d’époques.

{table(full_metrics)}

{table(full_synth)}

L’autoencodeur complet n’a détecté aucun des cinq événements injectés dans ce test.
Il ne peut donc pas être recommandé comme détecteur opérationnel sur la base de
cette expérience. Les seuils quantiles et la reconstruction globale d’une fenêtre
peuvent manquer des altérations locales ou modérées. Le robuste repère quatre
événements, mais son F1 horaire faible montre que ce rappel ne suffit pas à garantir
des alertes précises. Une évolution future doit se baser sur une nouvelle validation
et réserver un nouveau test, plutôt que régler les seuils avec les résultats ci-dessus.
"""
if (quick / "injection_cases.csv").exists():
    cases = pd.read_csv(quick / "injection_cases.csv")
    text += "\n\n## Détail des événements injectés — mode rapide\n\n" + table(cases.loc[cases.split == "test"]) + "\n"
(root / "docs/RAPPORT.md").write_text(text, encoding="utf-8")
