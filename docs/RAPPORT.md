# Rapport comparatif — exécution réelle

Environnement Windows, Python 3.13.5, CPU, dépendances exactes dans
`requirements.lock.txt`. Résultats ci-dessous issus de `artifacts/quick/run.json`
et des fichiers CSV produits par le pipeline, sans métrique simulée.

## Données et protocole

Archive UCI : 2075259 lignes minute, 25979 mesures manquantes,
0 doublons ; 438 heures insuffisamment renseignées
sur l’archive complète. Période : 2006-12-16 17:00:00 à 2010-11-26 21:00:00.
SHA256 : `9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff`.

Mode rapide : 8760 heures de 2009, 5256 train, 1752 validation,
1752 test ; 73 heures complétées. Prévision 24h, 28 origines glissantes
dans le test, cibles imputées exclues des métriques. Dénominateur MASE train :
0.596960 kW. Les origines peuvent se recouvrir ; les résultats
ne constituent pas 672 erreurs indépendantes. Le test ne sert à aucun réglage.

## Prévision sur données originales

| model | MAE | RMSE | MASE |
| --- | --- | --- | --- |
| ETS | 0.6960 | 0.9216 | 1.1658 |
| Naïve | 0.9508 | 1.2453 | 1.5928 |
| Ridge direct | 0.5227 | 0.7025 | 0.8757 |
| Saisonnière 24h | 0.6726 | 0.9499 | 1.1268 |

Le meilleur MAE mesuré est celui de **Ridge direct**, 0.5227 kW,
soit 45.0% de réduction face à la naïve sur ces origines.
Cette observation sur un seul split ne prouve pas une supériorité générale.
Le résultat test ne modifie pas les hyperparamètres sélectionnés sur validation.

Sélection Ridge : [{'alpha': 1.0, 'validation_MAE': 0.4612544521202221}, {'alpha': 100.0, 'validation_MAE': 0.4557533781551846}, {'alpha': 1000.0, 'validation_MAE': 0.4539665433385122}]. Sélection historique ETS : [{'days': 14, 'validation_MAE': 0.42810030078140315}, {'days': 28, 'validation_MAE': 0.4083387169077642}].
Les erreurs par horizon sont dans `horizon_metrics.csv`, celles selon le week-end
dans `period_metrics.csv`. Les modèles sont évalués sur les mêmes origines.

Ridge : MAE à horizon 1h = 0.5616 kW,
à 24h = 0.6774 kW ;
horizon le plus difficile = 24h (0.6774 kW).
Une erreur croissante avec l’horizon n’est pas garantie : le cycle journalier peut
rendre certaines heures intermédiaires plus difficiles que 24h.

| model | weekend | MAE | RMSE | MASE |
| --- | --- | --- | --- | --- |
| ETS | False | 0.6222 | 0.7939 | 1.0423 |
| ETS | True | 0.9038 | 1.2110 | 1.5139 |
| Naïve | False | 0.8314 | 1.0880 | 1.3927 |
| Naïve | True | 1.2874 | 1.6078 | 2.1567 |
| Ridge direct | False | 0.4634 | 0.6049 | 0.7763 |
| Ridge direct | True | 0.6899 | 0.9237 | 1.1556 |
| Saisonnière 24h | False | 0.6673 | 0.9264 | 1.1178 |
| Saisonnière 24h | True | 0.6877 | 1.0133 | 1.1520 |

Graphiques interactifs autonomes : [série](../artifacts/quick/series.html),
[erreurs par horizon](../artifacts/quick/horizons.html),
[comparaison synthétique](../artifacts/quick/synthetic.html).

## Détection sur anomalies injectées uniquement

| split | method | precision | recall | F1 | PR_AUC_AP | event_recall | mean_delay_h_detected | false_positive_hours | events_detected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | Robuste | 0.3500 | 0.2211 | 0.2710 | 0.2288 | 0.8000 | 6.2500 | 39 | 4 |
| calibration | Autoencodeur | 0.0667 | 0.0632 | 0.0649 | 0.0773 | 0.2000 | 18.0000 | 84 | 1 |
| test | Robuste | 0.2353 | 0.2105 | 0.2222 | 0.1808 | 0.8000 | 12.2500 | 65 | 4 |
| test | Autoencodeur | 0.0813 | 0.1789 | 0.1118 | 0.0607 | 0.6000 | 0.3333 | 192 | 3 |

Les seuils sont fixés sur validation originale (quantile 99%) : {'Robuste': 6.492597169174072, 'Autoencodeur': 0.9523245316743851}.
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
  PyTorch CPU ; meilleur MAE mesuré de 0.523 kW sur le test 2009 retenu.
- Réalisation d’une application Streamlit en français avec rejeu causal, imports et
  exports CSV, évaluation d’anomalies injectées et tests de non-utilisation du futur.


## Mode complet : résultats distincts

Exécution effectuée sur 34588 heures : 20752 train,
6918 validation, 6918 test. 40 époques,
40 origines de validation et 100 origines test ; 437 heures imputées.
La première heure incomplète est exclue. Les partitions ne sont pas celles de 2009 :
on ne peut attribuer une différence de performance uniquement au nombre d’époques.

| model | MAE | RMSE | MASE |
| --- | --- | --- | --- |
| ETS | 0.5522 | 0.7404 | 0.7974 |
| Naïve | 0.6951 | 0.9365 | 1.0038 |
| Ridge direct | 0.4663 | 0.6103 | 0.6733 |
| Saisonnière 24h | 0.5348 | 0.7704 | 0.7723 |

| split | method | precision | recall | F1 | PR_AUC_AP | event_recall | mean_delay_h_detected | false_positive_hours | events_detected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | Robuste | 0.1354 | 0.1368 | 0.1361 | 0.0634 | 0.6000 | 12.6667 | 83 | 3 |
| calibration | Autoencodeur | 0.0000 | 0.0000 | 0.0000 | 0.0150 | 0.0000 | — | 72 | 0 |
| test | Robuste | 0.1802 | 0.2105 | 0.1942 | 0.1294 | 0.8000 | 8.7500 | 91 | 4 |
| test | Autoencodeur | 0.0000 | 0.0000 | 0.0000 | 0.0174 | 0.0000 | — | 44 | 0 |

L’autoencodeur complet n’a détecté aucun des cinq événements injectés dans ce test.
Il ne peut donc pas être recommandé comme détecteur opérationnel sur la base de
cette expérience. Les seuils quantiles et la reconstruction globale d’une fenêtre
peuvent manquer des altérations locales ou modérées. Le robuste repère quatre
événements, mais son F1 horaire faible montre que ce rappel ne suffit pas à garantir
des alertes précises. Une évolution future doit se baser sur une nouvelle validation
et réserver un nouveau test, plutôt que régler les seuils avec les résultats ci-dessus.


## Détail des événements injectés — mode rapide

| split | method | kind | detected | delay_h |
| --- | --- | --- | --- | --- |
| test | Robuste | pic | True | 0.0000 |
| test | Robuste | chute | False | — |
| test | Robuste | niveau | True | 15.0000 |
| test | Robuste | derive | True | 21.0000 |
| test | Robuste | constant | True | 13.0000 |
| test | Autoencodeur | pic | True | 0.0000 |
| test | Autoencodeur | chute | True | 1.0000 |
| test | Autoencodeur | niveau | True | 0.0000 |
| test | Autoencodeur | derive | False | — |
| test | Autoencodeur | constant | False | — |
