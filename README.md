# Détection des tweets sexistes (EXIST 2024)

Projet de NLP, INSA Lyon 5A (GALLAPONT Thibault, MALLE Arnaud, MUEL Thomas) : classification de tweets anglais et espagnols comme sexistes ou non (tâche 1 du challenge [CLEF EXIST 2024](https://nlp.uned.es/exist2024/)). La mesure est le F1 de la classe YES sur le dev (934 tweets avec étiquette).

Le rapport complet est [rapport/rapport.pdf](rapport/rapport.pdf), au format Springer Nature.

## Résultats principaux

| Méthode | Notebook | F1 YES (dev) |
| --- | --- | --- |
| TF-IDF + régression logistique | `Modelling.ipynb` | 0,781 |
| TF-IDF amélioré (étiquettes soft, seuil choisi en validation croisée) | `Modelling2.ipynb` | 0,789 |
| « Modèle 1 » : TF-IDF + features manuelles + prior du mot-clé + prior kNN (stacking, sans fine-tuning) | `Modelling2.ipynb` | 0,841 |
| Twitter-XLM-R base, moyenne hard + soft | `Modelling.ipynb` | 0,846 |
| Twitter-XLM-R large, soft, moyenne de 5 seeds (**peu de calcul**) | `Modelling2.ipynb` | 0,878 |
| Qwen3-14B fine-tuné en LoRA | `Modelling3.ipynb` | 0,892 |
| Ensemble Qwen3-14B + Qwen2.5-7B + XLM-R large (**beaucoup de calcul**) | `Modelling3.ipynb` | **0,896** |
| Qwen3.5-0.8B, poids édités sans entraînement (0,0022 % des poids) | `wsp/` | 0,724 (base : 0,680) |

## Organisation du dépôt

| Élément | Contenu |
| --- | --- |
| `Modelling.ipynb` | Phase 1 (avant la présentation) : exploration des données, TF-IDF, étiquettes soft, augmentation par traduction, fine-tuning de Twitter-XLM-R. |
| `Modelling2.ipynb` | Phase 2 : protocole en validation croisée à 5 folds, preprocessing corrigé, ablations (multi-classes, multi-tâches, AEDA, CNN), « premier modèle » bon marché, LLM (few-shot, justifications, LoRA, distillation), combinaisons. |
| `Modelling3.ipynb` | Phase 3 : LLM 14B en LoRA bf16 (Qwen2.5, Qwen3), XLM-R XL, ensembles et stacking ; modèle final. |
| `data/` | Train et dev d'EXIST 2024 ; `EXIST2024_training_2.json` contient en plus les traductions (créé par `Modelling.ipynb`). |
| `results/` | Tableaux de résultats (`resultats_modelling2.csv`, `resultats_modelling3.csv`) et prédictions sur le dev de tous les modèles (`dev_predictions_v2.csv`, `dev_predictions_v3.csv`). |
| `wsp/` | *Weight-space programming* de Qwen3.5-0.8B, sans entraînement, mené avec Claude Opus : script d'édition `edit.py`, scripts de diagnostic, journal des tentatives et rapport détaillé [wsp/REPORT.md](wsp/REPORT.md). |
| `rapport/` | Sources LaTeX du rapport, figures, et scripts qui les produisent (`make_figures.py` pour la phase 1, `make_figures_phase2.py` pour la phase 2). |
| `presentation/` | Supports de la présentation orale. |

## Lancer les notebooks

- **`Modelling.ipynb`** : Python 3.10 ou plus récent ; la première cellule installe les librairies. RoBERTa et la traduction ont besoin de PyTorch ≥ 2.6, de préférence avec un GPU NVIDIA :
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`.
  Au premier lancement, le notebook traduit le train et crée `data/EXIST2024_training_2.json`, ensuite rechargé directement.
- **`Modelling2.ipynb` et `Modelling3.ipynb`** : prévus pour Google Colab avec un grand GPU (RTX PRO 6000, 96 Go). Le projet est lu depuis Google Drive (`MyDrive/TweetClassification`) et chaque fold est mis en cache dans `cache_modelling2/` : une session interrompue reprend où elle s'était arrêtée. Comptez 7 à 8 h de GPU pour `Modelling2` et 6 à 7 h pour `Modelling3` ; les paramètres en tête de notebook permettent de couper les parties les plus longues.
- **`wsp/`** : voir la partie « Reproduire » de [wsp/REPORT.md](wsp/REPORT.md). Depuis la racine : `python wsp/edit.py` écrit le modèle édité dans `edited_model/` (ignoré par git).

## Régénérer le rapport

```bash
python rapport/make_figures.py          # figures de la phase 1
python rapport/make_figures_phase2.py   # figures de la phase 2 (lit results/ et wsp/)
cd rapport && latexmk -pdf rapport.tex
```
