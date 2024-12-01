# Système d'Authentification Faciale

Ce projet implémente un système d'authentification faciale de haute précision utilisant des techniques avancées de vision par ordinateur et d'apprentissage profond.

## Fonctionnalités

- Enregistrement d'un visage autorisé via webcam ou fichier image
- Authentification faciale avec un taux de faux positifs inférieur à 0,1%
- Reconnaissance robuste face aux variations de lumière et de distance
- Interface utilisateur simple en ligne de commande

## Prérequis

- Python 3.7+
- Webcam (pour l'enregistrement via webcam)

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/YoannDev90/Face-ID.git
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
## Utilisation

Exécutez le script principal :
```bash
python facial_auth.py
```
Suivez les instructions à l'écran pour enregistrer un nouveau visage ou effectuer une authentification.

## Note de Sécurité

Ce système est conçu pour un usage personnel ou éducatif. Pour une utilisation en production, des mesures de sécurité supplémentaires sont recommandées.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
