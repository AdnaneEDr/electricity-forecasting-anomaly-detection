# Démonstration de cinq minutes

1. **0:00–0:45** — Présenter le foyer UCI, l’unité kW, l’horizon 24h et l’absence
   d’annotations réelles. Ouvrir l’onglet Exploration, choisir une semaine.
2. **0:45–1:30** — Montrer le profil horaire et la STL, en précisant son usage
   rétrospectif. Expliquer la séparation chronologique et la complétion causale.
3. **1:30–2:30** — Dans Prévisions, comparer les quatre modèles, MAE et erreurs
   par horizon. Montrer une origine historique avec observations et exporter le CSV.
4. **2:30–3:45** — Dans Alertes, déplacer le curseur du rejeu historique ; choisir
   les deux détecteurs, comparer scores/seuils et événements. Expliquer la fenêtre
   se terminant au présent et les comportements suspects à examiner.
5. **3:45–4:30** — Choisir Injection test, montrer les F1 et rappels événement,
   expliquer pourquoi ces scores ne prouvent pas une performance sur anomalies réelles.
6. **4:30–5:00** — Importer `artifacts/quick/series.csv`, confirmer kW, exporter
   les alertes et citer les limites : foyer unique, fuseau inconnu, manque de labels.

Questions entretien : pourquoi Ridge plutôt qu’un LSTM ? Comparaison interprétable
et rapide CPU ; pourquoi pas découpage aléatoire ? Il expose le futur ; pourquoi MASE ?
Erreur ramenée à une échelle saisonnière calculée sur train ; pourquoi traîne après un
pic pour l’autoencodeur ? Le pic reste 48h dans les fenêtres de reconstruction.
