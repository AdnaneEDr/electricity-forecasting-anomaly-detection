# Vérifications exécutées

Le 13 octobre 2026, sous Windows avec Python 3.13.5, environnement isolé CPU.

| Étape | Résultat vérifié |
| --- | --- |
| Acquisition UCI | HTTP réussi, ZIP lu, SHA256 conservé |
| Schéma et préparation | 2 075 259 lignes, 25 979 mesures manquantes, aucun doublon ni gap minute ; 438 heures insuffisantes |
| Scripts de préparation | `python prepare.py` exécuté, CSV préparé généré |
| Mode rapide | Quatre prévisionnistes et deux détecteurs entraînés/évalués sur 2009 |
| Mode complet | Même pipeline sur 34 588 heures, 40 époques autoencodeur, 100 origines test |
| Rechargement | Ridge, ETS et state_dict PyTorch rechargés et comparés |
| Absence de futur | Tests de modification du futur et de calcul sur préfixe historique réussis, deux modes |
| Injections | Cinq types, copies distinctes de calibration/test, graines 43/44, métriques réellement calculées |
| Réévaluation | `evaluate.py` exécuté dans les deux modes, métriques identiques aux artefacts figés |
| Inférence/export | `infer.py` exécuté : 24 prévisions futures, CSV de scores et alertes générés |
| Streamlit | AppTest : modes original/injection/complet, navigation, curseur de rejeu, deux exports proposés |
| Import CSV | CSV valide injecté dans le widget en test ; fichier invalide refusé avec message ; tests de doublons/fréquence/valeurs |
| Serveur local | Démarrage sur 127.0.0.1:8517 ; endpoint santé HTTP 200, contenu `ok` |
| Qualité | 13 tests pytest réussis ; Ruff réussi ; `pip check` sans incompatibilité |

Les téléchargements du navigateur sont vérifiés par la présence des boutons et leur
génération dans AppTest ; les fichiers CSV sont aussi générés réellement par la CLI.
Il n’y a pas de vérification visuelle par capture d’écran ni de test sur compteur en direct.
Le rejeu est historique, à observation horaire déjà reçue.

Les commandes Linux et le workflow GitHub sont fournis mais n’ont pas été exécutés
sur Linux ou sur GitHub ici. L’index officiel PyTorch CPU utilisé dans les commandes
Linux/CI répond et contient la version verrouillée ; l’installation Linux n’a pas
été réalisée. La configuration CPU évite de dépendre d’un GPU.

Les réexécutions pendant le développement ont servi à corriger le code, les tests et
la sauvegarde. Aucun hyperparamètre ni seuil n’a été sélectionné à partir des labels
test. Aucune métrique n’est inventée. Le faible F1 synthétique et l’échec de détection
de l’autoencodeur complet sont conservés et analysés dans le rapport.
