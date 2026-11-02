# Détection des tweets sexistes (EXIST 2024)

Projet de NLP, INSA Lyon 5A : classification de tweets anglais et espagnols comme sexistes ou non (tâche 1 du challenge [CLEF EXIST 2024](https://nlp.uned.es/exist2024/)).

Tout le travail est dans `Modelling.ipynb` : modèles TF-IDF, étiquettes soft, augmentation par traduction et fine-tuning de RoBERTa.

## Données

Le dossier `data/` contient le train et le dev d'EXIST 2024 (tweets). Au premier lancement, le notebook traduit le train (anglais vers espagnol et espagnol vers anglais) et crée `EXIST2024_training_2.json`, qui contient les tweets originaux et leurs traductions. Ce fichier est ensuite rechargé directement.

## Lancer le notebook

- Python 3.10 ou plus récent ; la première cellule installe les librairies.
- RoBERTa et la traduction ont besoin de PyTorch ≥ 2.6, de préférence avec un GPU NVIDIA :
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`
