# Fiche dataset

Source : [Individual Household Electric Power Consumption, UCI](https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption).
Attribution : Georges Hebrail et Alice Berard (2006), DOI 10.24432/C58K54,
licence [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Un foyer à Sceaux, France ; décembre 2006 à novembre 2010. Mesures à la minute.
La variable utilisée est `Global_active_power`, puissance active moyenne en kW.
Ce n’est pas une énergie en kWh. L’agrégation est une moyenne horaire de puissance.
Le fuseau n’est pas spécifié par la source : aucune conversion UTC ni hypothèse
Europe/Paris imposée, et donc aucune correction fiable des heures d’été/hiver.

Le fichier contient des mesures manquantes et ne fournit pas d’étiquettes d’anomalies.
`data/metadata.json` contient les comptes effectivement mesurés, la période exacte,
les doublons et l’empreinte SHA256 de l’archive. La fréquence est régularisée par
resampling. Les éventuels doublons minute sont moyennés explicitement. Une heure
avec moins de 45 minutes mesurées est déclarée manquante, puis complétée avec la
dernière valeur connue, avec indicateur `imputed` conservé. La première heure peut
être incomplète et manquante ; le mode complet commence à la première heure valide.
Les cibles imputées sont exclues des métriques de prévision. Les alertes en zone
imputée peuvent être des artefacts et nécessitent inspection.

Le mode rapide utilise toute l’année 2009 : 8760 heures, saisonnalités quotidiennes
et hebdomadaires, et une amplitude annuelle visible. Un seul cycle annuel ne suffit
pas à apprendre robustement une saisonnalité annuelle ; aucun terme annuel n’est
imposé. Le mode complet utilise les presque quatre ans disponibles.

Hypothèses : valeurs majoritairement normales dans le train, habitudes suffisamment
stables pour un transfert vers validation/test, et mesures disponibles à la fin de
l’heure. Aucun point extrême n’est supprimé. L’évaluation synthétique n’établit pas
la précision réelle des alertes. La représentativité d’un foyer unique est limitée.
