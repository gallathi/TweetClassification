# Programmer les poids de Qwen3.5-0.8B pour détecter le sexisme, sans entraînement

Reproduction, sur la tâche 1 d'EXIST 2024, de l'approche « Weight-Space Programming » (Takehi et al., MIT CSAIL, 2026) : améliorer un modèle en **écrivant directement de nouvelles valeurs dans ses poids**, à partir de statistiques d'activations, sans gradient ni entraînement.

## Résumé

|                                         | Base       | Édité                         |
| --------------------------------------- | ---------- | ----------------------------- |
| F1 macro, validation (596 tweets)       | 0,561      | **0,725**                     |
| F1 macro, **test** (934 tweets)         | 0,595      | **0,727** (+0,132, IC95 [+0,098 ; +0,166]) |
| F1 YES, test                            | 0,680      | **0,724** (+0,044, IC95 [+0,014 ; +0,074]) |
| AUC, test                               | 0,670      | **0,801**                     |
| Part de réponses « Oui », test          | 72 %       | 50 % (49 % de YES réels)      |
| Poids modifiés                          | –          | **18 369 sur 853 M = 0,0022 %** |

- **Deux éditions suffisent.** La première est une écriture de rang 1, *creuse*, dans le MLP de la couche 13 : 16 colonnes de `down_proj`, soit 16 381 poids. La seconde réaligne les deux lignes `Oui` et `Non` de la tête de sortie (1 988 poids).
- **Ce qu'on a trouvé.** Le modèle *sait* mieux qu'il ne répond. Son résidu sépare les tweets sexistes des autres avec une AUC de 0,75, et 16 neurones de la couche 13 atteignent 0,79. Pourtant, sa décision Oui/Non s'appuie sur une direction presque orthogonale à cette information (cosinus ≈ 0,04) et elle est biaisée vers « Oui » (72 % des réponses).
- **Les garde-fous sont respectés.** La génération reste cohérente (KL du token suivant ≤ 0,024 nat), le modèle ne répond pas « Oui » à tout, et l'effet sur des questions Oui/Non générales reste faible. Il n'est toutefois pas nul (voir la partie 7).
- **Limite principale.** Le résultat reste loin des modèles fine-tunés de `rapport/rapport.pdf`, qui atteignent un F1 YES d'environ 0,85 sur le même dev, contre 0,724 ici.

---

## 1. Règles et protocole

**Règles respectées.**
- Aucune rétropropagation, aucun gradient, aucun optimiseur, ni LoRA, ni fine-tuning, ni classifieur entraîné.
- Tout le code tourne sous `torch.no_grad()`, avec `torch.set_grad_enabled(False)` dans `wsp/common.py`.
- Les seules opérations utilisées sont des passes avant, des hooks, des moyennes, des différences de moyennes, des écarts-types, des projections, une SVD (diagnostic), un système linéaire 2×2 et des produits extérieurs de rang 1.
- Le modèle original n'est jamais modifié sur disque : le modèle édité est écrit dans `./edited_model/`.

**Données.** EXIST 2024, tâche 1, à partir de `data/EXIST2024_training.json` et `data/EXIST2024_dev.json` (le fichier `_training_2` n'est pas utilisé). On reprend le protocole de `rapport/rapport.pdf` :

| Jeu        | Origine                         | Tweets (égalités 3-3 écartées) | YES   | Rôle                                                       |
| ---------- | ------------------------------- | ------------------------------ | ----- | ---------------------------------------------------------- |
| calib      | 90 % du train                   | 5 468                          | 44 %  | calcul des statistiques d'activations (moyennes) et choix des hyperparamètres |
| validation | 10 % du train (seed 42, même tirage que `rapport/rapport.pdf`) | 596              | 44 %  | jugement de chaque tentative                               |
| test       | dev EXIST 2024                  | 934                            | 49 %  | **un seul passage**, à la fin                              |

L'étiquette est le vote majoritaire : YES si au moins 4 annotateurs sur 6 répondent « oui ». Les hyperparamètres (a, c, t, δ) sont choisis sur **calib**, qui contient 5 468 tweets. La validation sert de juge indépendant.

**Prompt fixe.** Le tweet est inséré dans le gabarit de chat de Qwen3.5, avec le raisonnement (`thinking`) désactivé :

```
Texte : « {tweet} »

Ce texte est-il sexiste ? Réponds Oui ou Non.
```

La prédiction compare `logit(Oui)` et `logit(Non)`, où `Oui` et `Non` sont les tokens sans espace 234746 et 7865. Le score `s = logit(Oui) − logit(Non)` est calculé en float32 à partir du dernier état caché, car les logits bf16 sont quantifiés par pas de 0,125. Le modèle répond « Oui » si `s > 0`.

**Matériel.** GPU RTX 3050 Laptop (4 Go), torch 2.11 avec CUDA 12.8, transformers 5.22, modèle en bf16. Une passe sur calib prend environ 3 minutes.

## 2. Le modèle

`Qwen3_5ForConditionalGeneration` : 853 M paramètres, dont 752 M pour le texte et le reste pour l'encodeur visuel. La partie texte compte 24 couches hybrides, avec 3 couches *Gated DeltaNet* (attention linéaire) pour 1 couche d'attention complète. La dimension cachée est 1 024 et les MLP SwiGLU ont 3 584 neurones. Le RMSNorm est centré sur zéro (`x·(1+w)`).

Point important : **`tie_word_embeddings = true`**. La tête de sortie et la table d'embeddings sont la même matrice. Modifier les lignes `Oui` et `Non` de la tête change donc aussi la façon dont le modèle *lit* ces deux tokens en entrée. Les mots ` Oui` et ` Non` de la consigne sont d'autres tokens (avec espace) et ne sont pas touchés.

## 3. Base de référence

| Validation | F1 macro | F1 YES | Précision | Rappel | AUC   | % Oui | Matrice [[VN, FP], [FN, VP]] |
| ---------- | -------- | ------ | --------- | ------ | ----- | ----- | ---------------------------- |
| Base       | 0,561    | 0,625  | 0,510     | 0,807  | 0,660 | 70 %  | [[127, 205], [51, 213]]      |

Le modèle suit bien le format : « Oui » et « Non » réunissent environ 93 % de la probabilité du premier token. En revanche, il répond « Oui » beaucoup trop souvent.

## 4. Diagnostic

### 4.1 Les erreurs

- **Biais de classe.** Le modèle répond « Oui » à 70 % des tweets et à **63 % des tweets non sexistes**. Le seul choix d'un seuil sur calib ferait passer le F1 macro de 0,561 à 0,604.
- **Le modèle détecte le sujet, pas le sexisme.** Sur calib, le score moyen est plus élevé pour les tweets qui *rapportent* ou *dénoncent* le sexisme (REPORTED 0,40, JUDGEMENTAL 0,39) que pour ceux qui *sont* sexistes (DIRECT 0,31). Les faux positifs parlent de genre, de patriarcat, de pronoms ou de sexualité, ou contiennent des insultes non genrées. EXIST compte REPORTED et JUDGEMENTAL comme sexistes, ce qui maintient le rappel. Le problème est que *tout* tweet sur les femmes est poussé vers « Oui ».
- **L'espagnol est moins bien traité** (AUC 0,60 contre 0,73 en anglais). Beaucoup de faux négatifs sont des sexismes implicites en espagnol : « Las mujeres no deben opinar », « niñata », « Se nota que es mujer por la envidia ».
- **Le score suit l'accord des annotateurs, mais faiblement.** Il vaut en moyenne 0,07 pour les tweets à 0 « oui » sur 6, et 0,39 pour ceux à 6 sur 6.

### 4.2 Où les classes se séparent (résidu, couche par couche)

On capture le résidu à la position de réponse après chaque couche. Pour chaque couche, on calcule sur calib `d = μ(sexiste) − μ(non sexiste)`, puis l'AUC sur val de la projection `h·d`. Aucun classifieur n'est entraîné.

| Couche (sortie) | 1    | 5    | 8    | 10   | 12   | **13** | 16   | 20   | 22   | 23 (finale) |
| --------------- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ---- | ---- | ----------- |
| AUC val de `h·d`  | 0,56 | 0,55 | 0,63 | 0,69 | 0,74 | **0,75** | 0,73 | 0,75 | 0,75 | 0,74 |
| cos(`d`, `W[Oui]−W[Non]`) | −0,05 | −0,04 | −0,00 | −0,01 | −0,03 | 0,01 | 0,02 | 0,04 | 0,06 | 0,07 |

- L'information « sexiste » se construit entre les couches 7 et 13, puis reste stable (AUC ≈ 0,75).
- **La direction de lecture Oui/Non est presque orthogonale à cette information** (|cos| ≤ 0,10 à toutes les couches). Le modèle ne lit pas ce qu'il sait : sa sortie n'atteint qu'une AUC de 0,66.
- Combiner plusieurs couches n'apporte rien (0,751 au mieux), et retirer les composantes principales de `d` dégrade les résultats. Une simple différence de moyennes plafonne donc à environ 0,75 dans le résidu.

### 4.3 Seize neurones de la couche 13

Les activations intermédiaires des MLP, `act(gate·x) ⊙ up·x` avec 3 584 neurones, plafonnent elles aussi à une AUC de 0,757 avec la différence de moyennes dense, à la couche 13. En revanche, si l'on **ne garde que les neurones de plus grande taille d'effet** `|μ₁ − μ₀| / σ`, l'AUC sur val **monte à 0,79-0,80 avec 8 à 16 neurones**. Le signal est concentré dans quelques neurones et noyé dans le bruit quand on prend tous les neurones.

Les 16 neurones retenus sont 500, 907, 945, 1041, 1045, 1067, 1082, 1440, 1959, 2179, 2213, 2393, 2509, 2826, 2839 et 3072. Leurs tailles d'effet vont de 0,74 à 0,96. Ensemble, ils forment un **détecteur de discours sur le genre et la sexualité** :
- **Les tweets qui les activent le plus** parlent de phallocentrisme, de fessée, de « feminazi », de culture du viol ou de biologie et d'oppression des femmes, et sont majoritairement JUDGEMENTAL.
- **Ceux qui les activent le moins** sont des offres d'emploi, des scores de football ou des publicités.
- L'activation moyenne de la clé vaut 0,08 pour les tweets YES et 0,025 pour les NO, dans les deux langues. Sa corrélation avec le nombre de « oui » des annotateurs est de 0,56.
- Projetée sur l'unembedding, leur sortie d'origine ne mène à rien de lisible : ces neurones n'écrivaient pas leur détection vers Oui/Non.

## 5. L'édition retenue (`wsp/edit.py`)

**A. Rang 1 creux dans le MLP de la couche 13** (`down_proj`, 16 colonnes, 16 381 poids)

```
a  = activations intermédiaires du MLP 13 à la position de réponse (calib)
k  = μ(a | sexiste) − μ(a | non) restreinte aux 16 neurones de plus grande |k_i|/σ_i, normalisée
u  = μ(r | sexiste) − μ(r | non), r = résidu en sortie de la couche 13, normalisée
down_proj ← down_proj + γ · u kᵀ          γ = t · ‖μ_r1 − μ_r0‖ / (kᵀ(μ_a1 − μ_a0)),  t = 4
```

Quand les « neurones du sexisme » s'activent, le MLP écrit en plus la direction sexiste dans le flux résiduel, avec 4 fois l'écart de classes déjà présent. Les couches 14 à 23 transmettent et amplifient ce signal. Avec la seule lecture d'origine, l'AUC de calib passe de 0,63 à 0,75.

**B. Lignes `Oui`/`Non` de la tête de sortie, spécifiques à la tâche** (2 lignes, 1 988 poids)

```
h   = état final normalisé à la position de réponse (après A) ; d = μ(h | sexiste) − μ(h | non)
m_t = moyenne de h sur la tâche (calib) ; m_g = moyenne de h sur 40 questions Oui/Non générales
b_t, b_g ∈ vect{m_t, m_g} tels que b_t·m_t = 1, b_t·m_g = 0 et b_g·m_t = 0, b_g·m_g = 1 (système 2×2)
Δ   = d/‖d‖² − (d/‖d‖² · m_g) · b_g                   # direction sans effet moyen hors tâche
W[Oui] += Δ/2 − c·b_t/2 + δ·b_t ;  W[Non] += −Δ/2 + c·b_t/2 + δ·b_t
```

- **Δ** réaligne la lecture sur la direction qui sépare réellement les classes : l'écart des scores moyens entre classes augmente de 1.
- **c = 0,51** est le biais qui maximise le F1 macro sur calib. Il ne s'applique que dans le contexte de la tâche : le décalage moyen mesuré est de −0,85 sur la tâche et de **−0,003 sur les questions générales**.
- **δ = 0,27** relève ensemble `Oui` et `Non` sur la tâche, pour qu'ils restent la réponse générée (argmax sur 99,2 % de calib).

**Proportion de poids modifiés : 18 369 / 852 985 920 = 0,00215 %** (0,00244 % des poids texte). C'est sept fois moins que les 0,016 % de l'article. Le détail est dans `edited_model/edit_meta.json`.

## 6. Journal des tentatives

Les mesures portent sur la validation (596 tweets). Pour E1 à E5, la tête de sortie est ré-estimée sur 2 000 tweets de calib après chaque édition intermédiaire (`wsp/experiments.py`). Les journaux bruts sont dans `wsp/journal_v0.jsonl` et `wsp/journal.jsonl`. Pour situer les écarts, l'erreur-type du F1 sur 596 tweets est d'environ 0,02.

| #   | Hypothèse                                                                 | Modification                                                     | Poids modifiés      | F1 macro val | AUC val | Conclusion |
| --- | ------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------------------- | ------------ | ------- | ---------- |
| 0   | –                                                                         | modèle de base                                                   | 0                   | 0,561        | 0,660   | répond Oui à 70 % |
| H-off | La lecture n'utilise pas l'information du résidu (hors ligne, sur cache) | grille de `a` sur la différence de moyennes finale + biais       | –                   | 0,688 (a=1)  | 0,744   | plateau dès a = 1 |
| H-off | Le bruit commun masque le signal                                        | `d` sans les top-k composantes principales (SVD)                 | –                   | 0,60-0,67    | ≤ 0,743 | **rejetée** : l'information est dans les composantes de forte variance |
| H-off | Une lecture creuse suffit                                               | Δ réduit à ses k plus grandes coordonnées                         | –                   | 0,647 (k=16) à 0,684 (k=512) | ≤ 0,744 | la version dense (1 024) reste la meilleure |
| E1 v0 | Biais de classe                                                       | biais le long de `m/‖m‖²`, dense                                  | 623 (0,00007 %)     | 0,602        | 0,660   | l'arrondi bf16 efface une partie du biais (55 % de Oui) |
| E1  | Idem, biais précis                                                        | biais écrit sur les 8 coordonnées les plus stables de `h`         | **16**              | 0,600        | 0,660   | le biais seul donne +0,04 |
| E2  | Réaligner la lecture                                                      | `W[Oui/Non] ± d/2‖d‖²` + biais                                    | 2 028 (0,00024 %)   | 0,678        | 0,743   | **+0,117** (IC95 [+0,077 ; +0,157]) |
| E3  | Amplifier la direction sexiste au milieu du réseau                        | `down_proj ← (I+βd̂d̂ᵀ)down_proj`, couche 13 (β=2 ; 8), couche 20 (β=4) | 3,4-3,6 M (0,40 %) | 0,678 / 0,692 / 0,680 | ≤ 0,751 | gain marginal pour un coût énorme |
| E4  | « Si clé sexiste, écrire la direction sexiste » (style ROME, dense)       | `down_proj += γ û k̂ᵀ`, couche 13 (γ=0,5 ; 2), couche 22 (γ=1)    | 2,9-3,4 M (0,34 %)  | 0,701 / 0,699 / 0,667 | ≤ 0,765 | dépasse le plafond linéaire (0,745) à la couche 13 |
| E5  | Idem, **clé creuse** (16 neurones de plus forte taille d'effet)           | 16 colonnes de `down_proj[13]`, t = 0,5 / 1 / 2 / 4 / 8 / 16      | 18 400 (0,0022 %)   | 0,689 / 0,706 / 0,721 / 0,719 / **0,733** / 0,723 | 0,761 → 0,802 → 0,774 | F1 de calib maximal à t = 8 (0,737) ; plateau |
| F1  | `edit.py`, E5 avec t = 8 et B globale, sur tout calib                     | A + B globale                                                    | 18 298              | 0,738        | 0,802   | **effet de bord** : questions factuelles 0,77 → 0,57, `**` en tête de 14 % des réponses |
| F2  | Rendre B spécifique à la tâche (système 2×2 avec `m_t`, `m_g`) et corriger le format | A (t = 8) + B spécifique                              | 18 410              | 0,737        | 0,800   | questions factuelles 0,73, format 100 % ; A pousse vers Oui sur les questions générales (67 %) |
| **F3** | Réduire l'écriture dans le MLP (même support)                          | **A (t = 4) + B spécifique : édition finale**                     | **18 369 (0,00215 %)** | **0,725**    | **0,798** | écart non significatif avec F2 (−0,012, IC95 [−0,028 ; +0,005]) ; effets de bord plus faibles, KL divisée par 4 |

**Arrêt.** Les gains plafonnent. Sur le balayage de t, le F1 de calib vaut 0,703, 0,717, 0,723 et 0,737 pour t = 1, 2, 4 et 8, puis redescend à 0,730 pour t = 16. Combiner plusieurs couches n'apporte rien hors ligne. Le dernier ajustement (F2 → F3) ne cherchait pas à améliorer le score mais à **réduire les effets de bord**, conformément à la consigne de garder l'édition la plus petite possible.

**Contribution de chaque édition** (bootstrap apparié sur val, 2 000 tirages) :
- la lecture seule (E2) apporte **+0,117** par rapport à la base ;
- l'édition du MLP ajoute **+0,059** (IC95 [+0,026 ; +0,092]) par rapport à E2, avec t = 8.

## 7. Garde-fous

Le script est `wsp/guardrails.py`, et les sorties complètes sont dans `wsp/guardrails.json`.

**La génération reste cohérente.** En génération greedy sur 5 prompts généraux (photosynthèse, poème, capitale de l'Argentine, recette de crêpes, 17 × 23), les textes du modèle édité sont aussi fluides et pertinents que ceux de la base. Ils diffèrent dans leur formulation, mais pas en qualité.

| Mesure | Base vs édité (t = 4) | Pour comparaison (t = 8) |
| --- | --- | --- |
| KL du token suivant, texte général FR/EN/ES | 0,005 / 0,024 / 0,005 / 0,007 nat | 0,019 / 0,095 / 0,022 / 0,029 |
| Même token le plus probable que la base | 90-100 % | 81-100 % |

**Le modèle ne répond pas « Oui » à tout.**
- Réponse réellement générée (argmax sur tout le vocabulaire) pour 100 tweets de val : 50 Oui, 50 Non, **0 autre**. La base donnait 63 Oui et 37 Non.
- Sur le test, le modèle répond « Oui » à 50 % des tweets, pour 49 % de YES réels. Le taux de faux positifs passe de 61 % à 28 %.
- Phrases non sexistes écrites à la main : 7 sur 8 sont classées Non. L'exception est « Women deserve equal pay #EqualPay », que la base classait déjà Oui. Phrases sexistes explicites : 3 sur 4 sont classées Oui. « Calladita estás más guapa » reste classée Non, de justesse (−0,15).

**Effet sur des questions Oui/Non générales** (`wsp/factual_gen.py`, 30 questions factuelles absentes du jeu de référence, réponses générées) :

| Exactitude | « Réponds Oui ou Non. » | « … puis justifie en une phrase. » |
| --- | --- | --- |
| Base | 0,77 | 0,77 |
| Édité, B globale (F1) | 0,57 | – |
| Édité, A t = 8 + B spécifique (F2) | 0,70 | 0,77 |
| **Édité final (F3)** | **0,73** | **0,87** |

L'édition finale ne dégrade pas ces questions de façon mesurable (±1 à 3 questions sur 30, dans le bruit). Certaines réponses individuelles changent malgré tout : « L'eau bout-elle à 100 °C au niveau de la mer ? Réponds Oui ou Non, puis justifie » donne « **Non**, l'eau ne bout pas… », alors que la base répondait « **OUI** ».

## 8. Évaluation finale (un seul passage sur le test)

| Jeu        | Modèle | F1 macro | F1 YES | Précision | Rappel | AUC   | % Oui | Matrice [[VN, FP], [FN, VP]] |
| ---------- | ------ | -------- | ------ | --------- | ------ | ----- | ----- | ---------------------------- |
| Validation | Base   | 0,561    | 0,625  | 0,510     | 0,807  | 0,660 | 70 %  | [[127, 205], [51, 213]]      |
| Validation | Édité  | **0,725**| **0,707** | 0,672  | 0,746  | **0,798** | 49 % | [[236, 96], [67, 197]]   |
| **Test**   | Base   | 0,595    | 0,680  | 0,569     | 0,844  | 0,670 | 72 %  | [[188, 291], [71, 384]]      |
| **Test**   | Édité  | **0,727**| **0,724** | **0,714** | 0,734 | **0,801** | 50 % | [[345, 134], [121, 334]] |

**Par langue (test)** :

| Langue | F1 macro base → édité | AUC base → édité |
| --- | --- | --- |
| Anglais | 0,617 → 0,747 | 0,727 → 0,837 |
| Espagnol | 0,564 → **0,708** | 0,609 → **0,776** |

**Rappel des YES par intention (test)** : DIRECT 0,84 → 0,74, REPORTED 0,81 → 0,68, JUDGEMENTAL 0,88 → 0,77. On échange une part du rappel contre beaucoup de précision (0,57 → 0,71).

Les résultats du test (0,727) sont cohérents avec ceux de la validation (0,725) : le choix de l'édition n'a pas sur-ajusté la validation.

**Pour situer ces chiffres.** Sur le même dev, les modèles fine-tunés de `rapport/rapport.pdf` atteignent un F1 YES d'environ 0,85 (XLM-R, BERTweet), et TF-IDF avec régression logistique 0,781. Avec 0,724, le modèle édité dépasse nettement sa propre base, mais reste en dessous de toute méthode supervisée. Ce résultat n'est pas surprenant : l'édition n'exploite que des moyennes de classes, là où un fine-tuning ajuste des millions de paramètres.

## 9. Ce que nous avons appris sur le modèle

1. **Le modèle sait mieux qu'il ne répond.** L'information « sexiste ou non » est présente dès la couche 10 et culmine à la couche 13, mais la direction lue par `Oui/Non` lui est presque orthogonale (cos ≈ 0,04). Réaligner deux lignes de la tête de sortie (2 048 poids) fait déjà passer le F1 macro de 0,56 à 0,68.
2. **Le défaut principal est un biais de calibration propre à la tâche.** Face à un texte qui parle de femmes, le modèle répond « Oui » par défaut (63 % de faux positifs). Ce biais n'existe pas sur des questions factuelles : la base y est équilibrée. Il faut donc le corriger *dans le contexte de la tâche seulement*, faute de quoi on dégrade toutes les questions Oui/Non (0,77 → 0,57). Le plan `{m_t, m_g}` fournit une solution en forme fermée.
3. **L'information utile est concentrée dans quelques neurones.** 16 neurones sur 3 584 du MLP 13 séparent mieux les classes (AUC 0,79) que tout le résidu (0,75). La version dense de la même édition modifie 160 fois plus de poids pour un résultat inférieur (0,34 % contre 0,0022 %).
4. **Ces neurones détectent un discours sur le genre et la sexualité, pas le sexisme lui-même.** Ils réagissent le plus aux tweets qui *dénoncent* le sexisme. Cela convient à EXIST, qui compte ces tweets comme sexistes, mais rend le modèle incapable de distinguer un tweet sexiste d'un tweet qui le dénonce.
5. **Les écritures au milieu du réseau sont exploitées de façon non linéaire.** Écrire la direction sexiste dans le résidu de la couche 13 fait monter l'AUC finale au-delà du plafond linéaire (0,80 contre 0,745). Les couches suivantes renforcent donc le signal injecté.

## 10. Limites

- **L'écart avec le supervisé reste important** : F1 YES de 0,724 contre environ 0,85 pour un modèle fine-tuné. Les différences de moyennes ne capturent qu'une direction par édition.
- **Les étiquettes servent quand même.** Les directions sont des moyennes par classe calculées sur 5 468 tweets annotés. Ce n'est pas un apprentissage itératif, mais c'est une forme de supervision. Les hyperparamètres (a, c, t, n, la couche) sont choisis sur calib, et le passage de t = 8 à t = 4 s'appuie aussi sur les garde-fous et la validation.
- **Des choix restent arbitraires.** La couche 13 est choisie au vu du diagnostic, n = 16 a priori, et le jeu de 40 questions de référence est écrit à la main. D'autres choix donneraient d'autres chiffres, dans le bruit de la validation (±0,02).
- **L'édition est liée au prompt.** Elle est calibrée pour la formulation « Ce texte est-il sexiste ? Réponds Oui ou Non. ». Sur une autre formulation ou un autre format de réponse (« Yes/No »), le gain n'est pas garanti.
- **Les embeddings sont liés.** Les lignes `Oui` et `Non` modifiées servent aussi d'embeddings d'entrée. Ces deux lignes changent de 5 % (`Oui`) et 6 % (`Non`) en norme relative, ce qui modifie un peu la lecture de ces tokens (sans espace) en entrée. Toutes les autres lignes sont identiques.
- **Les effets de bord ne sont pas nuls.** Sur 30 questions factuelles l'exactitude ne baisse pas de façon mesurable, mais certaines réponses individuelles basculent. L'édition A modifie un peu la distribution de *tous* les tokens (KL ≤ 0,024 nat), et le style des textes générés change sans perte visible de qualité.
- **La tête MTP n'est pas sauvegardée.** transformers ne charge pas la tête de prédiction multi-token (`mtp.*`). Elle n'est donc ni utilisée, ni modifiée, ni réécrite dans `edited_model/`, ce qui explique sa taille de 1,706 Go contre 1,747 Go pour l'original.
- **Les erreurs systématiques demeurent.** Le modèle ne sépare toujours pas un tweet sexiste d'un tweet qui dénonce le sexisme (ce qu'EXIST n'exige pas). Il rate encore des sexismes implicites (« Calladita estás más guapa ») et ne gère pas l'ironie.
- **La variance n'est pas estimée.** Il n'y a qu'un découpage et qu'une graine. Les intervalles de confiance donnés sont des bootstraps sur les tweets, pas sur les choix de l'expérience.

## 11. Reproduire

```bash
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install transformers accelerate safetensors scikit-learn pandas numpy huggingface_hub

.venv/Scripts/python wsp/edit.py                # écrit ./edited_model/ (~7 min sur RTX 3050)
cd wsp
../.venv/Scripts/python guardrails.py         # garde-fous de génération (base vs édité)
../.venv/Scripts/python factual_gen.py        # 30 questions factuelles, A/B séparés
../.venv/Scripts/python final_eval.py         # val + test, base et édité (une seule fois)
```

Variantes du journal : `wsp/edit.py --t 8` (F2), `wsp/edit.py --t 8 --readout global` (F1), `wsp/edit.py --t 0` (lecture seule). Les scripts de diagnostic et d'expérimentation sont dans `wsp/` : `collect.py`, `diagnose.py`, `readout_offline.py`, `capture_mlp.py` et `experiments.py`.

| Fichier | Rôle |
| --- | --- |
| `wsp/edit.py` | applique l'édition au modèle original et écrit `edited_model/` et `edit_meta.json` |
| `wsp/common.py` | données, découpage, prompt, score Oui/Non, métriques |
| `wsp/generic_questions.py` | 40 questions Oui/Non de référence (pour `m_g`) |
| `wsp/journal*.jsonl` | journal brut des tentatives E1 à E5 |
| `wsp/final_results.json` | résultats finaux val et test, par langue et par intention |
| `wsp/guardrails.json` | sorties des garde-fous |
