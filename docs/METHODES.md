# Méthodes et prévention des fuites

## Prévision

Une origine `o` est la première heure à prévoir. Les entrées proviennent strictement
de `[0,o)`, les 24 cibles de `[o,o+24)`. Train, validation et test sont contigus
(60/20/20), sans mélange aléatoire. Les fenêtres train se terminent avant la limite
train. Les contextes de validation/test peuvent contenir le passé des partitions
précédentes et les observations test déjà reçues : c’est une évaluation glissante,
pas une prévision unique de toute la partition.

La naïve répète la dernière valeur. La naïve saisonnière répète les 24 dernières
heures. ETS additif quotidien sans tendance évite une extrapolation instable ; son
historique 14 ou 28 jours est choisi sur validation puis il est réajusté uniquement
sur le passé à chaque origine. Ridge multi-sorties estime directement les 24
horizons, sans récursion. Entrées : 24 dernières valeurs, 24 valeurs d’une semaine
plus tôt, dernière valeur, moyenne/écart-type sur 24h, moyenne sur 168h et sinus/cosinus
de l’heure et du jour de semaine connus à l’origine. Aucune météo future inconnue.
StandardScaler est appris sur les seules fenêtres train.

Références officielles vérifiées : [Ridge scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
et [ExponentialSmoothing statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html).

Ridge teste alpha 1, 100, 1000 avec deux blocs temporels et train croissant ; les
transformations de chaque fold restent dans son train. Après sélection, le modèle
livré reste ajusté au train initial, comme les détecteurs. Le test n’est pas utilisé
pour régler les modèles. Les origines test sont espacées et le nombre est fixé par
le mode. MAE et RMSE sont en kW ; MASE utilise la moyenne des erreurs saisonnières
à retard 24 sur train, jamais sur test. Une MASE supérieure à 1 signifie que
l’erreur dépasse cette échelle train, sans comparaison à la baseline sur le même
test. Aucun intervalle de prévision n’est revendiqué.

## Alertes

Le détecteur robuste calcule `y[t]-y[t-24]`, centré et réduit par médiane et MAD
du train pour l’heure correspondante. L’échelle minimale 0,05 kW empêche une division
par une dispersion nulle. Le score est une valeur absolue ; il peut repérer à la
fois augmentations et diminutions.

Autoencodeur dense 48→24→8→24→48, ReLU et Adam : faible coût CPU, deux journées de
contexte, goulot d’étranglement empêchant une simple copie parfaite. Le train est
supposé majoritairement normal mais peut contenir des événements suspects. Son
scaler est appris sur train seulement. Les mini-batches train sont mélangés pour
l’optimisation, sans modifier le découpage chronologique.

La MSE de reconstruction de chaque fenêtre est affectée uniquement à son dernier
horodatage. Il n’y a aucune redistribution sur les points précédents, qui utiliserait
alors leurs observations futures. Une alerte en ligne inclut l’observation courante,
reçue à la fin de l’heure, ainsi que les 47 heures précédentes. Le score peut rester
élevé après un événement sorti progressivement de la fenêtre. Cette traîne peut
réduire la précision horaire et doit être expliquée en entretien.

Chaque seuil est le quantile 99% des scores de validation originale. Cela fixe une
fréquence empirique en validation, pas une garantie de faux positifs sur test.
Les alertes consécutives forment un événement sans fusion des gaps. Une anomalie
injectée est détectée par événement si au moins une alerte apparaît pendant sa
durée. Le délai est le nombre d’heures entre le début injecté et la première alerte
dans l’événement ; les événements manqués sont exclus du délai moyen et comptés
dans le rappel événement. Aucun « point adjustment » avec les labels test.
PR-AUC est ici l’average precision (AP) scikit-learn, pas une intégrale trapézoïdale.

## Réalité, injection et rétrospective

Les deux détecteurs reçoivent exactement les mêmes cinq événements injectés dans
des copies distinctes : pic, chute, niveau, dérive et plateau constant. Positions,
durées, amplitudes et graines sont dans `run.json`. La calibration synthétique est
diagnostique, graine 43 sur validation ; test synthétique graine 44 sur test. Elle
ne règle pas les seuils, qui restent calculés sur validation originale. Les amplitudes
de test utilisent l’écart-type du passé avant test. Une seule réalisation ne mesure
pas la variabilité statistique et ne justifie pas une généralisation.

STL robuste sert uniquement à l’exploration rétrospective du train : ses lissages
utilisent les deux côtés du temps. Elle ne nourrit jamais le moteur d’alerte.
La simulation Streamlit transmet uniquement le préfixe reçu au détecteur. Des tests
modifient le futur et vérifient que caractéristiques, scores et alertes passés restent
identiques. `verification.json` vérifie aussi cette propriété sur les modèles appris.

Les tests d’interface utilisent [AppTest Streamlit](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
