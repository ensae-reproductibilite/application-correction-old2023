# Probabilité de survie sur le Titanic [![Construction image Docker](https://github.com/ensae-reproductibilite/application-correction/actions/workflows/prod.yml/badge.svg)](https://github.com/ensae-reproductibilite/application-correction/actions/workflows/prod.yml)


Découvrir de manière interactive le modèle sur [ce site](https://ensae-reproductibilite.github.io/application-correction/)
ou par le biais de l'[API](https://titanic.kub.sspcloud.fr/docs).

Pour pouvoir utiliser ce projet, il 
est recommandé de créer un fichier `config.yaml`
ayant la structure suivante:

```yaml
jeton_api: ####
train_path: ####
test_path: ####
test_fraction: ####
```

## Réutilisation

Pour pouvoir tester ce projet, le code suivant
suffit:

```python
pip install -r requirements.txt
python main.py
```