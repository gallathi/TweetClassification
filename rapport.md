# Détection des tweets sexistes (EXIST 2024, tâche 1) : rapport de projet

_GALLAPONT Thibault, MALLE Arnaud, MUEL Thomas. Projet de NLP, INSA Lyon 5A._

---

## Résumé

On cherche à décider si un tweet, en anglais ou en espagnol, est sexiste ou non. C'est la tâche 1 du challenge CLEF EXIST 2024. La mesure retenue est le **F1 de la classe YES**.

Le notebook principal, `Modelling.ipynb`, compare des modèles TF-IDF classiques à un transformer multilingue fine-tuné, **Twitter-XLM-R base**. Le meilleur résultat est **0,846 de F1 sur le dev**. On l'obtient en moyennant deux versions de Twitter-XLM-R : l'une entraînée sur le vote majoritaire, l'autre sur la proportion de « oui ». TF-IDF plafonne à 0,781.

On a ensuite étudié la littérature du challenge, puis testé une seconde stratégie :

- traduire tous les tweets en anglais ;
- utiliser des **BERTweet-large**, dont l'un est déjà spécialisé dans la détection du sexisme ;
- ajouter des **features manuelles**, l'augmentation **AEDA**, une tête **MultiCNN** et une sortie à 4 classes.

Sur une dizaine de configurations, le F1 reste **entre 0,833 et 0,848 sur le dev**. Or, avec 934 tweets, l'incertitude sur un F1 est d'environ ±2,5 points. **Aucune variante ne fait donc mieux que `Modelling.ipynb` de façon mesurable**, alors qu'elles coûtent environ trois fois plus cher en calcul.

Nos contraintes étaient les suivantes : classification binaire, modèles de la famille BERT uniquement, pas de LLM génératif, et un seul jeu de dev pour évaluer. Dans ce cadre, on considère avoir atteint le plateau de ce que nos données permettent de mesurer.

---

## 1. Le problème et les données

### 1.1 La tâche

EXIST 2024 fournit des tweets en anglais et en espagnol, chacun annoté par **six personnes**. Pour la tâche 1, chaque annotateur répond YES (le tweet est sexiste) ou NO.

**Point important : dans EXIST, un tweet qui rapporte ou dénonce du sexisme compte comme sexiste**, au même titre qu'un tweet lui-même sexiste. Les annotateurs qui répondent « oui » précisent l'intention du tweet (tâche 1.2) :

- **DIRECT** : le tweet est lui-même sexiste ;
- **REPORTED** : il rapporte un fait sexiste ;
- **JUDGEMENTAL** : il dénonce le sexisme.

À partir des six réponses, on construit deux étiquettes :

- **hard** : le vote majoritaire, avec au moins 4 « oui » sur 6 pour YES. Les égalités 3-3 sont écartées ;
- **soft** : la proportion de « oui », par exemple 4/6 ≈ 0,67. Elle garde la trace du désaccord entre annotateurs.

### 1.2 Les données et le protocole

|                                 | Train                                       | Dev               |
| ------------------------------- | ------------------------------------------- | ----------------- |
| Tweets                          | 6 920 (3 260 en anglais, 3 660 en espagnol) | 1 038 (489 / 549) |
| YES (hard)                      | 2 697 (39 %)                                | 455 (44 %)        |
| NO (hard)                       | 3 367 (49 %)                                | 479 (46 %)        |
| Égalités 3-3                    | 856 (12 %)                                  | 104 (10 %)        |
| Part de YES, anglais / espagnol | 40 % / 49 %                                 | 44 % / 53 %       |

Le protocole d'évaluation :

- **On entraîne sur le train et on évalue sur le dev**, qui sert de jeu de test. Les étiquettes du test officiel ne sont pas publiques.
- **10 % du train servent de validation**, toujours avec le même tirage (seed 42). Elle sert à choisir l'époque, le seuil ou le modèle, sans toucher au dev.
- **La mesure principale est le F1 de la classe YES**, celle qu'utilise EXIST. Elle est calculée sur les 934 tweets du dev qui ont une étiquette hard.

Le dev contient un peu plus de YES que le train, dans les deux langues. Le train et le dev ne partagent pratiquement aucun tweet : on n'a trouvé qu'un seul quasi-doublon.

### 1.3 Ce que l'analyse des données nous a appris

**Le désaccord entre annotateurs**

- **L'unanimité est rare.** Beaucoup de tweets reçoivent 2, 3 ou 4 « oui » sur 6, et les modèles se trompent surtout sur ces tweets ambigus. Les 856 égalités, soit 12 % du train, sont perdues avec les étiquettes hard. C'est ce qui justifie les étiquettes soft.
- **La sévérité varie énormément d'un annotateur à l'autre.** Le train compte 725 annotateurs, répartis en groupes de 6 qui annotent chacun environ 57 tweets. Selon l'annotateur, la part de « oui » va de **0,07 à 0,93**, et selon le groupe, de 0,21 à 0,63. Une partie de l'étiquette dépend donc du groupe qui a annoté le tweet, ce qui est du bruit pour le modèle.
- **Les informations sur les annotateurs ne sont pas exploitables.** Le genre et l'âge sont identiques pour tous les tweets d'une même langue, et les champs pays, ethnie et niveau d'études sont absents de nos fichiers. Par ailleurs, seuls 29 des 113 annotateurs du dev apparaissent dans le train. Les meilleures équipes d'EXIST tiraient pourtant une partie de leurs gains de ces informations (voir la partie 3).
- **Environ 55 % des votes « oui » sont REPORTED ou JUDGEMENTAL** : le tweet parle du sexisme sans être lui-même sexiste. C'est la principale difficulté du problème, car un tweet qui dénonce le sexisme emploie les mêmes mots qu'un tweet sexiste.

**Ce que le texte révèle.** Mesures faites sur les tweets du train qui ont une étiquette hard :

| Élément                                     | Part des tweets | Part de YES quand il est présent | Part de YES sinon |
| ------------------------------------------- | --------------- | -------------------------------- | ----------------- |
| URL                                         | 31 %            | **36 %**                         | 49 %              |
| Hashtag                                     | 14 %            | **32 %**                         | 47 %              |
| Le tweet commence par une mention (réponse) | 47 %            | 49 %                             | 41 %              |
| Emoji                                       | 18 %            | 41 %                             | 45 %              |
| Rires (« jaja », « lol »)                   | 2 %             | **60 %**                         | 44 %              |
| Mots censurés (« f\*ck »)                   | 0,3 %           | 70 %                             | –                 |

- **Les URL et les hashtags indiquent plutôt un tweet non sexiste.** Beaucoup sont des campagnes : 8 % de YES pour #WomensDay, 0 % pour #WeAreEqual. Mieux vaut donc les remplacer par un jeton que les supprimer.
- **Certains emojis sont très parlants** : 🙄 (73 % de YES), 🤷 (67 %), ❤ (17 %).
- **Les mots-clés de collecte biaisent les modèles.** _women_ (77 % de YES) et _mujeres_ (85 %) poussent un modèle à associer « parler des femmes » et « sexiste ».
- **Les tweets sont courts** : 177 caractères en moyenne, 795 au maximum. Seuls 0,2 % dépassent 160 tokens.
- **Il y a quelques doublons.** On compte 10 groupes de quasi-doublons (38 tweets), dont 3 aux étiquettes contradictoires. Par exemple, « Los hombres también sufren depresión postparto » apparaît 7 fois avec 4, 4, 2, 1, 1, 2 et 1 « oui ». Huit tweets ne contiennent qu'un message automatique (« account temporarily unavailable »).

---

## 2. Le notebook principal : `Modelling.ipynb`

Ce notebook s'exécute en local, sur un GPU RTX 3070 de portable. On commence par des modèles simples (TF-IDF et modèles linéaires) pour avoir une référence, puis on fine-tune RoBERTa, le type de modèle utilisé par les meilleures équipes du challenge.

### 2.1 Chargement et étiquettes

Les fichiers sont des dictionnaires JSON {id: tweet}, chargés dans des DataFrames. Pour chaque tweet, on compte le nombre de « oui » (`n_yes`), on en déduit l'étiquette soft (`p_yes`) et l'étiquette hard (`hard`, vide en cas d'égalité).

L'exploration donne trois constats :

- **Les deux langues sont à peu près aussi représentées.** On entraîne donc **un seul modèle pour les deux**, ce qui double la quantité de données.
- **Les classes sont assez équilibrées**, avec un peu plus de NO en anglais.
- **L'unanimité est rare**, d'où l'intérêt des étiquettes soft (partie 2.7).

### 2.2 Nettoyage

Le nettoyage est volontairement léger :

- les entités HTML sont décodées ;
- les URL deviennent `url` ;
- les mentions deviennent `@user`, pour ne pas apprendre des noms de comptes ;
- les hashtags sont découpés en mots selon les majuscules (#IDoNotRenounce devient « i do not renounce »), car ils sont souvent importants ;
- les lettres répétées sont réduites (« siiiii » devient « sii »).

**On ne retire pas les stopwords** : il faudrait deux listes, une par langue, et des mots comme _not_ ou _she_ peuvent compter.

On garde deux versions du texte : **clean**, en minuscules, pour TF-IDF, et **light**, avec les majuscules, pour RoBERTa, qui a été pré-entraîné sur du texte brut.

### 2.3 Mots indicateurs de sexisme

On ajoute au preprocessing une courte liste de mots, une par langue, dont l'emploi signale presque toujours un tweet sexiste :

- des insultes visant spécifiquement les femmes (_whore_, _zorra_, _marimacho_) ;
- des clichés (_dumb blonde_, _rubia_, _sumisa_) ;
- le mot _feminazi_.

**Construction de la liste :** on est parti des mots les plus associés à YES dans le **train**, puis on n'a gardé que ceux qui sont sexistes par leur sens et présents à 80 % ou plus dans des tweets sexistes. Des insultes comme _bitch_ ou _puta_ ont été écartées, car elles apparaissent aussi beaucoup dans des tweets non sexistes : insultes générales, citations, dénonciations. Le dev n'a pas servi à construire la liste.

**Utilisation :** quand un tweet contient l'un de ces mots, on ajoute à la fin le marqueur `lex_sexiste`. TF-IDF voit déjà chaque mot séparément, mais ces mots sont rares (une vingtaine d'apparitions chacun) et leurs poids sont donc peu fiables. Le marqueur les regroupe : le modèle apprend un seul poids, plus solide, pour tous ces mots. On ne l'utilise que pour TF-IDF : RoBERTa connaît déjà le sens de ces mots, et le marqueur serait pour lui un mot inconnu.

| Langue   | Tweets contenant un mot de la liste | Part de YES parmi eux | Part des tweets YES couverts |
| -------- | ----------------------------------- | --------------------- | ---------------------------- |
| Anglais  | 227                                 | 93 %                  | 18 %                         |
| Espagnol | 199                                 | 95 %                  | 12 %                         |

La liste est très fiable, mais elle ne couvre qu'une petite partie des tweets sexistes, car beaucoup ne contiennent ni insulte ni cliché évident. C'est donc un indice pour le modèle, pas une règle de décision.

### 2.4 Augmentation par traduction (désactivée)

On a environ 3 300 tweets en anglais et 3 700 en espagnol. Pour doubler les données dans les deux langues, on a traduit tout l'espagnol en anglais et tout l'anglais en espagnol, en gardant les annotations de chaque tweet : un tweet sexiste en espagnol donne aussi un tweet sexiste en anglais.

On a utilisé les modèles **MarianMT** d'Helsinki-NLP (`opus-mt-en-es` et `opus-mt-es-en`). Ce sont de petits modèles spécialisés chacun dans une paire de langues : ils sont gratuits, tournent en local et traduisent les 7 000 tweets en une dizaine de minutes sur GPU. Le résultat est enregistré dans `EXIST2024_training_2.json`. **On ne traduit que le train** ; sinon, on évaluerait le modèle sur des tweets qu'il aurait vus dans l'autre langue.

**Cela n'a rien apporté :**

- aucun gain pour TF-IDF ;
- −1,4 point de F1 pour RoBERTa hard ;
- +0,4 point pour RoBERTa soft.

Le modèle étant déjà multilingue, une copie traduite lui apprend peu de choses nouvelles. Les traductions gardent globalement le sens, mais certaines phrases sont traduites mot à mot et quelques tweets ne sont même pas en espagnol, ce qui ajoute du bruit. Cette partie, qui triple le temps d'exécution, est désactivée par défaut (`USE_TRANSLATION = False`).

### 2.5 Modèles TF-IDF

TF-IDF représente chaque tweet par les mots qu'il contient, pondérés selon leur rareté. On compare deux variantes :

- **mots** : mots seuls et paires de mots ;
- **mots + caractères** : on ajoute des morceaux de 2 à 5 lettres. Ils résistent mieux aux fautes, à l'argot et aux variantes (_feminazi_, _feminazis_), et fonctionnent dans les deux langues.

Chaque famille de modèles a une raison d'être :

- une **baseline** qui répond toujours la classe majoritaire, pour savoir à partir de quand un modèle apprend vraiment quelque chose ;
- trois **modèles linéaires**, en général les plus efficaces sur du TF-IDF, car ils gèrent bien des dizaines de milliers de variables creuses :
  - la **régression logistique**, qui donne des probabilités et dont les coefficients se lisent facilement ;
  - le **SVM linéaire**, référence classique de la classification de textes ;
  - le **Ridge Classifier**, une régression linéaire régularisée, très rapide ;
- **Naive Bayes** (variante Complement, adaptée au texte), le modèle le plus simple, souvent utilisé comme point de comparaison ;
- **Random Forest** et **XGBoost**, pour vérifier si des modèles non linéaires apportent quelque chose. Comme ils sont lents sur des vecteurs aussi grands, ils ne sont entraînés que sur la représentation par mots.

| Représentation    | Modèle                        | F1 (YES)  | Macro F1 |
| ----------------- | ----------------------------- | --------- | -------- |
| Mots + caractères | **Régression logistique**     | **0,781** | 0,792    |
| Mots + caractères | SVM linéaire                  | 0,775     | 0,785    |
| Mots + caractères | Ridge                         | 0,771     | 0,783    |
| Mots              | SVM linéaire                  | 0,762     | 0,774    |
| Mots              | Ridge                         | 0,757     | 0,770    |
| Mots              | Régression logistique         | 0,756     | 0,771    |
| Mots              | Naive Bayes                   | 0,737     | 0,741    |
| Mots              | XGBoost                       | 0,726     | 0,749    |
| Mots + caractères | Naive Bayes                   | 0,703     | 0,718    |
| Mots              | Random Forest                 | 0,694     | 0,735    |
| —                 | Baseline (classe majoritaire) | 0,000     | 0,339    |

- **La baseline a un F1 nul**, puisqu'elle ne prédit jamais YES : tous les autres modèles apprennent donc quelque chose.
- **Les trois modèles linéaires sont en tête**, autour de 0,78 avec les caractères. Ajouter les caractères leur fait gagner entre 1 et 2,5 points.
- **Naive Bayes et les modèles à arbres restent nettement derrière.** Sur du TF-IDF, la non-linéarité des arbres n'apporte rien et coûte beaucoup de temps de calcul.
- **On garde la régression logistique**, parce qu'elle donne des probabilités.
- **Ce score est légèrement optimiste** : le meilleur modèle a été choisi sur le dev, qui sert aussi à l'évaluer.
- **Les mots indicateurs n'améliorent pas TF-IDF** (0,778 contre 0,781, un écart trop petit pour être significatif). TF-IDF voit déjà chacun de ces mots, et ils ne concernent qu'une petite partie des tweets sexistes.

Par langue, la régression logistique obtient 0,768 en anglais et 0,790 en espagnol.

### 2.6 Analyse des erreurs (TF-IDF)

- **Les erreurs se concentrent sur les cas ambigus.** Le modèle se trompe sur 11 à 15 % des tweets où tout le monde est d'accord, mais sur près de 40 % de ceux qui n'ont reçu que 4 « oui » sur 6. Une bonne partie des erreurs porte donc sur des cas où les humains eux-mêmes hésitent.
- **Le modèle réagit à des mots, pas au sens.** Côté YES, les mots les plus influents sont _women_, _mujeres_, des insultes et le vocabulaire du féminisme. C'est logique, mais cela montre la limite de TF-IDF : un tweet qui dénonce le sexisme utilise souvent les mêmes mots qu'un tweet sexiste. C'est précisément ce qu'un transformer devrait mieux gérer.

### 2.7 Étiquettes soft

Plutôt que d'apprendre « ce tweet est sexiste », on apprend « 4 personnes sur 6 le trouvent sexiste ». Cela permet d'utiliser les tweets à égalité et devrait donner des probabilités plus réalistes sur les cas ambigus.

scikit-learn n'accepte pas directement des cibles entre 0 et 1. On contourne le problème en dupliquant chaque tweet : une fois étiqueté YES avec un poids `p_yes`, une fois NO avec un poids `1 − p_yes`. La régression logistique minimise alors exactement la cross-entropy avec la proportion d'annotateurs.

On compare avec le F1 et avec la **cross-entropy** entre la probabilité prédite et `p_yes`. Plus elle est basse, plus le modèle reproduit fidèlement l'avis des annotateurs.

| Modèle                  | F1 (YES) | Cross-entropy |
| ----------------------- | -------- | ------------- |
| TF-IDF, étiquettes hard | 0,781    | 0,614         |
| TF-IDF, étiquettes soft | 0,775    | **0,571**     |

Le F1 ne bouge presque pas, mais la cross-entropy baisse nettement : le modèle soft est moins catégorique sur les tweets ambigus, et ses probabilités collent mieux à l'avis des annotateurs.

### 2.8 RoBERTa

**Pourquoi un transformer.** Dans l'article de synthèse d'EXIST 2024, presque toutes les meilleures équipes fine-tunent un modèle de la famille RoBERTa. Contrairement à TF-IDF, RoBERTa lit la phrase entière et représente chaque mot dans son contexte. Il peut donc, en principe, distinguer « les femmes ne savent pas conduire » de « j'en ai marre qu'on dise que les femmes ne savent pas conduire ».

**Le modèle** : [`cardiffnlp/twitter-xlm-roberta-base`](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base), pour deux raisons :

- c'est un **XLM-RoBERTa**, la version multilingue de RoBERTa : un seul modèle suffit pour l'anglais et l'espagnol ;
- il a été ré-entraîné sur environ 200 millions de **tweets** : il connaît déjà l'argot, les hashtags et les emojis.

**Le fine-tuning.** On ajoute une couche de classification à 2 classes au-dessus du modèle, puis on ré-entraîne tout pendant quelques époques sur nos tweets.

- **La boucle d'entraînement est écrite à la main**, plutôt qu'avec le Trainer de Hugging Face : on voit ce qui se passe, et on peut facilement entraîner sur des cibles soft. La perte est la cross-entropy entre la sortie du modèle et [1 − p, p], où p vaut 0 ou 1 en hard et la proportion de « oui » en soft.
- **Les réglages** : 3 époques, AdamW avec un taux de 2e-5 et un weight decay de 0,01, warmup sur 10 % des pas puis décroissance linéaire, batch de 16, 128 tokens au plus, demi-précision.
- **On garde l'époque qui a la plus faible cross-entropy sur la validation.**
- **Deux versions** sont entraînées :
  - **hard** : 5 468 tweets, égalités retirées ;
  - **soft** : 6 236 tweets.
- **Durée** : une dizaine de minutes sur un GPU de portable pour les deux modèles, plusieurs heures sur CPU.

**Le sur-apprentissage.** Avec les étiquettes hard, la cross-entropy de validation remonte dès la 2e époque (0,735, puis 0,756, puis 0,959), alors que la perte d'entraînement continue de baisser. Le modèle devient trop sûr de lui, même sur les tweets ambigus, d'où l'intérêt de garder la meilleure époque. Avec les étiquettes soft, la courbe reste presque plate (0,556 à 0,558) : le modèle ne peut pas être plus catégorique que les annotateurs.

**La moyenne des deux versions.** Les versions hard et soft ne font pas exactement les mêmes erreurs, puisqu'elles ont appris sur des étiquettes différentes. En moyennant leurs probabilités, les erreurs propres à chaque version se compensent en partie. C'est un petit ensemble, qui ne demande aucun entraînement supplémentaire.

| Modèle                       | F1 (YES)  | Cross-entropy |
| ---------------------------- | --------- | ------------- |
| TF-IDF, hard                 | 0,781     | 0,614         |
| TF-IDF, soft                 | 0,775     | 0,571         |
| RoBERTa, hard                | 0,843     | 0,658         |
| RoBERTa, soft                | 0,834     | **0,533**     |
| **Moyenne des deux RoBERTa** | **0,846** | 0,541         |

- **RoBERTa fait nettement mieux que TF-IDF** : environ 0,84 de F1 contre 0,78, dans les deux langues (version hard : 0,833 en anglais, 0,850 en espagnol). Lire la phrase en contexte fait une vraie différence.
- **Hard ou soft** : la version hard a le meilleur F1 (0,843), la version soft les probabilités les plus fidèles aux annotateurs (0,533 contre 0,658).
- **La moyenne des deux obtient le meilleur F1 du notebook (0,846)**, avec des probabilités presque aussi fidèles que la version soft (0,541). Le gain reste faible : les deux versions partent du même modèle et ont vu les mêmes tweets, donc elles font en grande partie les mêmes erreurs.

### 2.9 Conclusion de `Modelling.ipynb`

- **TF-IDF et un modèle linéaire** donnent une base solide, autour de 0,78. Leur limite se voit dans les mots les plus influents : le modèle réagit à des mots comme _women_, sans comprendre le sens.
- **RoBERTa** monte à environ 0,84 en lisant les tweets en contexte, et à 0,846 en moyennant ses versions hard et soft.
- **Les étiquettes soft** changent peu le F1, mais donnent des probabilités plus fidèles au désaccord des annotateurs et limitent le sur-apprentissage.
- **Les mots indicateurs** n'apportent pas de gain : ils sont très fiables, mais couvrent peu de tweets, et TF-IDF voit déjà ces mots.
- **Une bonne partie des erreurs restantes** porte sur des tweets où les annotateurs eux-mêmes ne sont pas d'accord.

---

## 3. Exploration bibliographique

On a lu quelques papiers d'EXIST 2024-2025 et de SemEval-2023 pour voir les techniques utilisées par les meilleurs équipes. Le jeu de tweets d'EXIST 2025 est **identique** à celui d'EXIST 2024, donc les résultats 2025 sont directement comparables.

On s'est fixé trois contraintes : classification binaire uniquement, aucun LLM génératif (on reste sur la famille BERT), et la traduction autorisée seulement avec des modèles de traduction dédiés.

### 3.1 Ce que font les meilleures équipes

| Rang (tâche 1, hard) | Équipe          | F1 YES sur le test | Méthode                                                                                                                         |
| -------------------- | --------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| 1er en 2024          | NYCU-NLP        | 0,794              | DeBERTa-v3-large et XLM-R-large, rétro-traduction, AEDA, infos sur les annotateurs, multi-tâches, validation croisée en 5 folds |
| 2e en 2024           | ABCD Team       | 0,783              | un modèle par groupe d'annotateurs, puis agrégation                                                                             |
| 3e en 2024           | CIMAT-CS-NLP    | 0,790              | ensemble entre un LLM en zero-shot (Gemini) et des XLM-R fine-tunés                                                             |
| 4e en 2024           | EquityExplorers | 0,778              | espagnol traduit en anglais, fusion de RoBERTa-large et DeBERTa-v3-large                                                        |
| 1er en 2025          | Mario           | 0,817              | Llama-3.1-8B fine-tuné en QLoRA                                                                                                 |
| 2e en 2025           | CIMAT-GTO       | 0,800              | justifications générées par un LLM, ajoutées au tweet avant Twitter-XLM-R                                                       |

Le constat principal : toutes les équipes de tête utilisent un modèle _large_ ou un LLM. Deux équipes ont déjà traduit l'espagnol en anglais pour utiliser des modèles anglais : EquityExplorers, et DiTana-PV, avec le même BERTweet « sexisme » que nous, mais sur les memes.

### 3.2 Les idées retenues

- **Traduire vers l'anglais et utiliser un modèle anglais fort.** Chez NYCU-NLP, traduire l'espagnol en anglais améliore DeBERTa-v3-large, un modèle anglais, mais dégrade XLM-R-large, un modèle multilingue.
- **BERTweet-large.** Sur EDOS (SemEval-2023), il bat DeBERTa-v3-large et RoBERTa-large en macro-F1 (0,866 contre 0,860 et 0,840).
- **Partir d'un modèle déjà fine-tuné pour le sexisme** sur EDOS, puis le réentraîner sur EXIST.
- **AEDA** : on insère au hasard entre 1 et n/3 signes de ponctuation parmi `. ; ? : ! ,`, sans changer aucun mot. Sur cinq jeux de données, AEDA fait mieux qu'EDA (synonymes, insertions, permutations, suppressions).
- **La tête MultiCNN.** Chez Kalra & Zubiaga, elle fait passer BERT-base de 0,751 à 0,760 de F1 sur EXIST 2021, avec des convolutions de largeurs 4, 6 et 8. Dans la même étude, EDA fait _baisser_ le score (0,760 → 0,748).
- **La sortie à 4 classes** (NO, DIRECT, REPORTED, JUDGEMENTAL), dans l'esprit des modèles multi-tâches de GrootWatch et NYCU-NLP. De même, une équipe d'EXIST 2025 transforme le binaire en 7 classes (le nombre de « oui », de 0 à 6).
- **Des features manuelles** qui résument des signaux rares.
- **Une leçon de méthode** (AILS-NTUA, EXIST 2026) : une sélection de variables optimisée sur le dev ne s'est pas généralisée au test. Il faut faire ses choix sur la validation, pas sur le jeu d'évaluation.

### 3.3 Les pistes écartées

| Piste                                                                      | Gain rapporté                              | Pourquoi on ne la suit pas                                                                                                      |
| -------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Justifications « pour » et « contre » générées par Llama 3 (CIMAT-GTO)     | F1 de validation de 0,829 à 0,859          | contrainte « pas de LLM »                                                                                                       |
| LLM fine-tuné en QLoRA (Mario)                                             | meilleur score 2025                        | contrainte « pas de LLM », au moins 16 Go de mémoire GPU                                                                        |
| Envoyer les cas incertains à un LLM (Warwick, « When in Doubt »)           | +1 à 1,5 point                             | contrainte « pas de LLM »                                                                                                       |
| Contexte web récupéré par un agent LLM (GrootWatch)                        | +4 points d'ICM sur le dev                 | LLM de 70 milliards de paramètres et recherche web                                                                              |
| Indicateurs extraits par un LLM et gradient boosting (AILS-NTUA)           | gain sur les memes                         | contrainte « pas de LLM » ; mauvaise généralisation du dev au test                                                              |
| Infos démographiques sur les annotateurs (NYCU-NLP, GrootWatch)            | +4 à 7 points d'ICM                        | absentes de nos fichiers                                                                                                        |
| Une tête de classification par annotateur (Davani et al.)                  | au moins aussi bon que le vote majoritaire | 725 annotateurs, environ 57 tweets chacun, presque tous différents entre train et dev                                           |
| EDA (synonymes, insertions…)                                               | négatif chez Kalra & Zubiaga               | AEDA fait mieux sans toucher aux mots                                                                                           |
| Pré-entraînement supplémentaire sur les 2 millions de textes d'EDOS        | gains à SemEval-2023                       | le BERTweet « sexisme » a déjà été fine-tuné sur EDOS ; coût élevé                                                              |
| Minuscules, suppression de la ponctuation, stopwords et stemming pour BERT | –                                          | font perdre de l'information à un modèle pré-entraîné qui distingue les majuscules ; la ponctuation est aussi la matière d'AEDA |
| Débruitage statistique des votes (modèle de Dawid-Skene)                   | inconnu                                    | le dev est évalué sur le vote majoritaire brut, donc rien ne garantit un gain                                                   |

---

## 4. Seconde stratégie : traduction en anglais et BERTweet (notebook `modelling2`)

Les expériences ont tourné sur Google Colab, lancé depuis VS Code, avec un GPU G4 (NVIDIA RTX PRO 6000 Blackwell).

### 4.1 Les choix de conception

**Le modèle « sexisme » et l'écart de définition.** Nous avions deux candidats, tous deux des BERTweet-large fine-tunés sur EDOS :

- [`NLP-LTU/bertweet-large-sexism-detector`](https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector) : macro-F1 de 0,861 sur EDOS, 10e sur 84 équipes ;
- [`tum-nlp/bertweet-sexism`](https://huggingface.co/tum-nlp/bertweet-sexism) : macro-F1 de 0,85.

On a retenu celui de NLP-LTU : l'ordre de ses classes est documenté et son score est un peu meilleur. Ce modèle n'est pas utilisable tel quel, car **EDOS ne définit pas le sexisme comme EXIST**. EDOS compte comme sexiste _« any abuse, implicit or explicit, that is directed towards women based on their gender »_, c'est-à-dire un abus. EXIST compte en plus les tweets qui rapportent ou dénoncent le sexisme, soit environ la moitié des votes « oui ». Il faut donc **le réentraîner sur EXIST**, et le comparer au même modèle sans spécialisation, [`vinai/bertweet-large`](https://huggingface.co/vinai/bertweet-large).

**Le preprocessing, adapté à chaque type de modèle :**

| Étape                                               | Pour BERTweet | Pour TF-IDF ou des embeddings statiques   | Pourquoi                                                                            |
| --------------------------------------------------- | ------------- | ----------------------------------------- | ----------------------------------------------------------------------------------- |
| Mentions → `@USER`, URL → `HTTPURL`, emojis → texte | oui           | oui                                       | c'est le format de pré-entraînement de BERTweet ; mentions et URL sont informatives |
| Réduire les suites de mentions et d'URL             | oui           | oui                                       | moins de tokens, moins de bruit                                                     |
| Normaliser les rires et les lettres répétées        | oui           | oui                                       |                                                                                     |
| Remplacer les tirets par des espaces                | oui           | oui                                       | découpe les mots composés                                                           |
| Garder les hashtags et les découper en mots         | oui           | oui                                       | la présence d'un hashtag est informative                                            |
| Mettre en minuscules                                | non           | oui                                       | BERTweet distingue les majuscules, qui marquent l'emphase                           |
| Retirer la ponctuation                              | non           | oui, sauf les apostrophes                 | elle porte du sens (« ? », guillemets ironiques) et AEDA s'appuie dessus            |
| Stopwords, stemming                                 | non           | à tester, en gardant négations et pronoms | ils cassent le découpage en sous-mots d'un transformer                              |

Pour BERTweet, on retire seulement ce qui n'apporte aucun contexte : le préfixe « RT », les suites de mentions et d'URL, et les 8 tweets réduits à un message automatique. Un petit dictionnaire découpe les hashtags écrits tout en minuscules, que le découpage par majuscules ne coupe pas (#yositecreo devient « yo sí te creo »). Les phrases collées sont séparées (« capullo.El » devient « capullo. El »). Les mots qu'on veut **amplifier** passent par les features manuelles, et non par une modification du texte.

**Le modèle de traduction (espagnol vers anglais) :**

| Modèle                                  | Taille     | Remarques                                                                                            |
| --------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------- |
| `Helsinki-NLP/opus-mt-es-en` (MarianMT) | 77 M       | utilisé dans `Modelling.ipynb` ; rapide, mais faible sur l'argot                                     |
| **`facebook/nllb-200-3.3B`**            | 3,3 B      | **retenu** : la plus grande version « dense » de NLLB, environ 7 Go de mémoire GPU en demi-précision |
| `google/madlad400-3b-mt`                | 3 B        | alternative, plus robuste aux fautes selon MT-LENS                                                   |
| `google/madlad400-7b-mt` / `10b-mt`     | 7 / 10,7 B | les plus gros modèles de traduction ouverts, disproportionnés pour 4 200 tweets                      |

Quelques précautions :

- **Le dev est traduit au moment de la prédiction.** Ce n'est pas une fuite de données, puisqu'on ne s'entraîne jamais dessus.
- **Les emojis sont retirés avant la traduction, puis remis à la fin du tweet**, car NLLB les supprime ou les déforme.
- **Les jetons `@USER` et `HTTPURL` sont réparés** quand la traduction les a altérés.
- **Le résultat est mis en cache**, pour ne traduire qu'une fois.

**Les 21 features manuelles.** Elles sont calculées sur le tweet original, avant traduction, avec des listes dans les deux langues. Elles ont été construites et mesurées sur le train uniquement, où le taux moyen de YES est de 44,5 % :

| Feature                                               | Part du train | Part de YES quand elle est présente |
| ----------------------------------------------------- | ------------- | ----------------------------------- |
| mots de la liste de `Modelling.ipynb`                 | 7,0 %         | 94 %                                |
| généralisation (« women are… », « las mujeres son… ») | 7,2 %         | 85 %                                |
| rôle traditionnel (kitchen, cocina…)                  | 0,6 %         | 84 %                                |
| vocabulaire du féminisme                              | 4,9 %         | 79 %                                |
| insulte genrée                                        | 5,4 %         | 77 %                                |
| référence aux femmes                                  | 31,0 %        | 72 %                                |
| vocabulaire du sexisme (sexism, machismo, acoso…)     | 10,1 %        | 71 %                                |
| emoji d'exaspération (🙄🤦🤷)                         | 1,7 %         | 64 %                                |
| corps, sexualité                                      | 3,6 %         | 63 %                                |
| référence aux hommes                                  | 18,2 %        | 58 %                                |
| hashtag de dénonciation (#MeToo…)                     | 1,5 %         | 57 %                                |
| hashtag, quel qu'il soit                              | 14,1 %        | 31 %                                |
| emoji cœur                                            | 1,6 %         | 31 %                                |
| hashtag de campagne (#WomensDay…)                     | 1,5 %         | 14 %                                |

Les autres features ont un signal plus faible : rires, URL, emojis de rire, question, guillemets, négation, réponse à un tweet. On a écarté les candidats sans signal : longueur du tweet, majuscules, points d'exclamation, pronoms à la première personne. Les comptages sont plafonnés à 3, puis les features sont centrées et réduites.

Dans le modèle, les features passent par une couche linéaire initialisée à zéro, dont les scores s'ajoutent à ceux calculés sur le texte. Pour la tête MultiCNN, elles sont concaténées au vecteur extrait du texte. Au départ, cette couche ne change donc rien, puis le modèle apprend le poids de chaque feature.

**Les autres ajouts :**

- **AEDA** : à chaque époque, on ajoute une copie de chaque tweet d'entraînement, avec de la ponctuation insérée au hasard.
- **La sortie à 4 classes** : la cible est la répartition des 6 votes entre NO, DIRECT, REPORTED et JUDGEMENTAL ; on revient au binaire avec P(YES) = 1 − P(NO). Pour le modèle sexisme, on élargit sa tête EDOS binaire à 4 classes. La ligne NO reprend « non sexiste », et les trois lignes YES reprennent « sexiste » avec un biais diminué de log 3. Au départ, le modèle donne donc exactement le même P(NO) qu'avant.
- **La tête MultiCNN** : des convolutions de largeurs 4, 6 et 8 sur les sorties de l'encodeur, suivies d'un max-pooling et de deux couches denses avec dropout.
- **Les taux d'apprentissage** : 1e-5 pour l'encodeur, 1e-4 pour la tête linéaire, 1e-3 pour la tête MultiCNN et la couche des features.

### 4.2 Essai 1 : comparaison des quatre modèles

Réglages : 3 époques, sortie à 4 classes, AEDA, époque choisie sur le F1 de validation. Dans cet essai, **seules les têtes MultiCNN recevaient les features**.

| Modèle                         | F1 validation | F1 dev    |
| ------------------------------ | ------------- | --------- |
| Sexisme + MultiCNN (features)  | 0,838         | **0,848** |
| Sexisme + linéaire             | **0,845**     | 0,847     |
| BERTweet + MultiCNN (features) | 0,843         | 0,846     |
| BERTweet + linéaire            | **0,845**     | 0,833     |

- **Les classements s'inversent entre validation et dev.** Sexisme + MultiCNN est dernier en validation et premier sur le dev. C'est le signe que ces écarts relèvent du bruit.
- **La tête MultiCNN n'apporte rien de mesurable** : −0,7 et −0,2 point en validation, +0,1 et +1,3 sur le dev.
- **Partir du modèle sexisme aide peut-être un peu** : +1,4 et +0,2 point sur le dev, mais rien en validation.
- **On retient Sexisme + linéaire** : c'est le seul modèle en tête sur les deux jeux, avec la tête la plus simple.

### 4.3 Essai 2 : entraînement plus long avec arrêt anticipé

Même modèle, cette fois avec les features, sur 10 époques au plus et avec arrêt après 2 époques sans progrès. Le F1 de validation a valu 0,836, puis 0,819, puis 0,827. L'entraînement s'est arrêté à la 3e époque et a gardé la 1re.

Résultat : **0,836 en validation, 0,848 sur le dev** (0,856 en anglais, 0,841 en espagnol traduit).

Cet essai était mal réglé :

- **Le calendrier du taux d'apprentissage était réparti sur 10 époques.** La 1re époque servait entièrement de montée en régime, et le taux n'avait pas encore commencé à baisser au moment de l'arrêt.
- **L'arrêt anticipé a réagi au bruit.** Sur 684 tweets de validation, le F1 varie d'environ un point d'une époque à l'autre.

### 4.4 Essai 3 : 4 époques complètes

Même modèle avec les features, sur 4 époques sans arrêt anticipé : le taux d'apprentissage descend jusqu'à 0. Le F1 de validation a valu 0,838, puis 0,844, puis 0,835, puis 0,831. La meilleure époque est la 2e ; ensuite, le modèle sur-apprend.

Résultat : **0,844 en validation, 0,839 sur le dev** (0,842 en anglais, 0,836 en espagnol traduit).

### 4.5 Ce que montre la traduction

NLLB garde en général le sens, mais commet des erreurs gênantes sur des tweets :

- « Lee sobre Gamergate » (« lis ») devient « Lee on Gamergate », comme un prénom ;
- « bancarrota » devient « BANK ROTA » ;
- « superen » (« passez à autre chose ») devient « override ».

Quelques tweets marqués espagnols sont en réalité en galicien (« Lourenzá organiza obradoiros de coeducación… »).

Sur le dev, l'espagnol traduit fait 0,6 à 1,5 point de moins que l'anglais (0,836 contre 0,842, et 0,841 contre 0,856). Dans `Modelling.ipynb`, l'espagnol non traduit faisait au contraire **mieux** que l'anglais (0,850 contre 0,833).

---

## 5. Synthèse des résultats

| Pipeline              | Modèle                                              | F1 validation | F1 dev    |
| --------------------- | --------------------------------------------------- | ------------- | --------- |
| `Modelling.ipynb`     | TF-IDF mots + caractères, régression logistique     | –             | 0,781     |
| `Modelling.ipynb`     | Twitter-XLM-R base, hard                            | –             | 0,843     |
| `Modelling.ipynb`     | Twitter-XLM-R base, soft                            | –             | 0,834     |
| `Modelling.ipynb`     | **Twitter-XLM-R base, moyenne hard + soft**         | –             | **0,846** |
| Traduction + BERTweet | BERTweet-large + linéaire                           | 0,845         | 0,833     |
| Traduction + BERTweet | BERTweet-large + MultiCNN + features                | 0,843         | 0,846     |
| Traduction + BERTweet | Sexisme + linéaire                                  | 0,845         | 0,847     |
| Traduction + BERTweet | Sexisme + MultiCNN + features                       | 0,838         | 0,848     |
| Traduction + BERTweet | Sexisme + linéaire + features, arrêt à la 3e époque | 0,836         | 0,848     |
| Traduction + BERTweet | Sexisme + linéaire + features, 4 époques            | 0,844         | 0,839     |

_Dans `Modelling.ipynb`, l'époque était choisie sur la cross-entropy de validation, et le F1 de validation n'est pas affiché._

**Pour situer ces scores :** les meilleurs F1 YES sur le **test** officiel étaient de 0,794 en 2024 (NYCU-NLP) et de 0,817 en 2025 (Mario, avec un LLM). Nos scores sont mesurés sur le **dev**, et ne sont donc pas comparables : toutes les équipes perdent plusieurs points entre leur validation et le test. CIMAT-GTO, par exemple, passe de 0,859 à 0,789.

---

## 6. Analyse critique

**Ce qui a fonctionné :**

- **Fine-tuner un transformer** : +6,5 points par rapport au meilleur TF-IDF. C'est le seul gain net du projet.
- **Les étiquettes soft** : des probabilités plus fidèles au désaccord des annotateurs et moins de sur-apprentissage, sans perte de F1.
- **La moyenne des versions hard et soft** : un petit ensemble qui ne demande aucun entraînement supplémentaire.

**Ce qui n'a pas fonctionné, ou pas de façon mesurable :**

- **La traduction**, que ce soit comme augmentation (MarianMT) ou pour utiliser des modèles anglais (NLLB). Elle ajoute des erreurs, alors que le modèle multilingue traite déjà très bien l'espagnol.
- **Les modèles _large_ anglais** : environ trois fois plus de calcul par tweet, plus le coût de la traduction, pour aucun gain mesurable.
- **La spécialisation EDOS** : au mieux un léger avantage, ce qui s'explique par l'écart de définition entre EDOS et EXIST.
- **La tête MultiCNN, les features manuelles, AEDA et la sortie à 4 classes** : aucun effet mesurable.
- **Entraîner plus longtemps** : le modèle sur-apprend dès la 2e ou la 3e époque, comme dans `Modelling.ipynb`.

**Les limites de notre évaluation :**

- **Un seul dev de 934 tweets.** L'intervalle de confiance à 95 % d'un F1 vaut environ ±2,5 points. Tous les écarts entre nos meilleurs modèles, 1,5 point au plus, sont dans cette marge.
- **Une seule seed par configuration.** Une partie des écarts vient du hasard de l'entraînement : initialisation, ordre des exemples, tirages d'AEDA. C'est aussi pourquoi les classements changent entre validation et dev.
- **Une validation de 684 tweets**, elle aussi bruitée, ce qui rend imprécis le choix de l'époque et du modèle.
- **Des comparaisons parfois imparfaites.** Dans l'essai 1, seules les têtes MultiCNN recevaient les features. L'essai 2 avait un calendrier mal réglé. Le GPU et la précision de calcul diffèrent entre `Modelling.ipynb` (RTX 3070, float16) et `modelling2` (G4, bfloat16 pour les essais 2 et 3).
- **Le choix du modèle TF-IDF a été fait sur le dev**, qui sert aussi à l'évaluer : son score est légèrement optimiste.

**Pourquoi on s'arrête là.** Avec nos contraintes, les pistes restantes promettent au mieux un point de F1, et ce gain ne serait pas mesurable sur notre dev. Les gains importants de la littérature viennent de LLM génératifs, que l'on a exclus, ou d'informations sur les annotateurs, que l'on n'a pas.

---

## 7. Conclusion et perspectives

**Le modèle retenu est celui de `Modelling.ipynb` : la moyenne de deux Twitter-XLM-R base, l'un entraîné sur les étiquettes hard, l'autre sur les étiquettes soft (0,846 de F1 sur le dev).** C'est le plus simple et le moins coûteux, il traite les deux langues sans traduction, et aucune des variantes testées ne fait mieux de façon mesurable.

Pistes non testées qui resteraient dans nos contraintes, avec un gain attendu faible :

- **plusieurs seeds par modèle, dont on moyenne les probabilités** : c'est la piste la plus sûre pour gagner un peu, et surtout pour mesurer de façon plus fiable ;
- **une validation croisée en 5 folds** à la place d'une seule validation de 10 %, pour choisir les réglages de façon moins bruitée ;
- **un modèle multilingue _large_** (XLM-RoBERTa-large, mDeBERTa-v3) sur les tweets originaux, comme NYCU-NLP en 2024 ;
- **un entraînement multi-tâches** qui prédit aussi les tâches 2 et 3, déjà annotées dans les données.

Pistes hors de nos contraintes, qui donnent les meilleurs résultats publiés : justifications générées par un LLM, fine-tuning d'un LLM, utilisation des informations démographiques des annotateurs.

---

## 8. Documentation

### Le challenge et les jeux de données

- **L. Plaza et al., _Overview of EXIST 2024 – Learning with Disagreement for Sexism Identification and Characterization in Tweets and Memes (Extended Overview)_, CLEF 2024.** https://ceur-ws.org/Vol-3740/paper-87.pdf
  Il décrit la tâche, les données, la mesure F1 YES et les approches de toutes les équipes : c'est notre référence pour situer nos résultats et repérer les méthodes efficaces.
- **L. Plaza et al., _Overview of EXIST 2025: Learning with Disagreement for Sexism Identification and Characterization in Tweets, Memes, and TikTok Videos_, CLEF 2025.** https://ceur-ws.org/Vol-4038/paper_135.pdf
  Il donne le classement 2025, sur les mêmes tweets qu'en 2024, et résume les tendances : étiquettes soft, rétro-traduction, AEDA, modèles multi-tâches.
- **H. R. Kirk et al., _SemEval-2023 Task 10: Explainable Detection of Online Sexism (EDOS)_, SemEval 2023.** https://aclanthology.org/2023.semeval-1.305
  Il présente le jeu EDOS, sur lequel le BERTweet « sexisme » a été entraîné, et sa définition du sexisme comme un abus, plus étroite que celle d'EXIST.

### Systèmes d'EXIST et de SemEval

- **Y.-Z. Fang, L.-H. Lee, J.-D. Huang, _NYCU-NLP at EXIST 2024: Leveraging Transformers with Diverse Annotations for Sexism Identification in Social Networks_.** https://ceur-ws.org/Vol-3740/paper-93.pdf
  C'est la recette du vainqueur 2024 (modèles _large_, AEDA, traduction, validation croisée en 5 folds) ; elle nous a guidés pour AEDA et la traduction vers l'anglais.
- **N. Nowakowski et al., _GrootWatch at EXIST 2025: Automatic Sexism Detection on Social Networks – Classification of Tweets and Memes_, CLEF 2025.** https://hal.science/hal-05157015
  Équipe de l'INSA Lyon (LIRIS) : elle donne des hyperparamètres pour XLM-R-large et montre l'intérêt des étiquettes soft et d'un modèle multi-tâches qui prédit aussi l'intention.
- **A. Kalra, A. Zubiaga, _Sexism Identification in Tweets and Gabs using Deep Neural Networks_, 2021.** https://arxiv.org/abs/2111.03612
  C'est la source de la tête MultiCNN et des têtes LSTM ou Bi-LSTM au-dessus de BERT, et elle montre qu'EDA fait baisser le score.
- **K. Chaviaras, M. Lymperaiou, A. Voulodimos, _Multimodal Sexism Identification and Characterization using Large Language Models and Gradient Boosting_ (AILS-NTUA), EXIST 2026.** https://arxiv.org/abs/2606.05997
  Ils combinent des indicateurs extraits par un LLM et du gradient boosting, et donnent une leçon de méthode : une sélection de variables faite sur le dev peut ne pas tenir sur le test.
- **S. S. Al-Azzawi et al., _NLP-LTU at SemEval-2023 Task 10: The Impact of Data Augmentation and Semi-Supervised Learning Techniques on Text Classification Performance on an Imbalanced Dataset_.** https://arxiv.org/abs/2304.12847
  C'est le papier du modèle `NLP-LTU/bertweet-large-sexism-detector` : BERTweet-large y bat DeBERTa-v3-large et RoBERTa-large, et la rétro-traduction n'aide pas les modèles _large_.
- **A. Rydelek, D. Dementieva, G. Groh, _AdamR at SemEval-2023 Task 10: Solving the Class Imbalance Problem in Sexism Detection with Ensemble Learning_.** https://aclanthology.org/2023.semeval-1.190
  C'est le papier du modèle `tum-nlp/bertweet-sexism`, l'autre candidat pour le modèle « sexisme ».
- **D. Chen, H. Qi, _Beyond Binary: 7-Class Sexism Identification via ModernBERT and SCL_, EXIST 2025.** https://ceur-ws.org/Vol-4038/paper_144.pdf
  Ils transforment le binaire en 7 classes (le nombre de « oui ») : c'est l'idée de départ de notre sortie à 4 classes.
- **A. M. Davani, M. Díaz, V. Prabhakaran, _Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations_, TACL 2022.** https://aclanthology.org/2022.tacl-1.6
  Ils proposent une tête de classification par annotateur pour tenir compte du désaccord ; la méthode est écartée chez nous faute de tweets par annotateur.
- **LCT-1** (https://arxiv.org/abs/2306.05075), **IUST_NLP** (https://arxiv.org/abs/2305.06892) et **Attention** (https://arxiv.org/abs/2304.04610), SemEval-2023.
  Ces trois équipes montrent qu'un pré-entraînement supplémentaire sur des textes du domaine aide ; dans notre cas, le modèle sexisme a déjà été fine-tuné sur EDOS.

### Approches à base de LLM (écartées par choix)

- **CIMAT-GTO, _Stacked Reflective Reasoning in Large Neural Language Models_, EXIST 2024.** https://ceur-ws.org/Vol-3740/paper-121.pdf
  Des justifications « pour » et « contre » générées par Llama 3, ajoutées au tweet avant le même Twitter-XLM-R que nous, font passer le F1 de validation de 0,829 à 0,859.
- **CIMAT-GTO, _Knowledge Expansion Guided by Justification for Improved Sexism Categorization_, EXIST 2025.** https://ceur-ws.org/Vol-4038/paper_167.pdf
  C'est la même idée, avec des justifications générées par Gemini : 2e en 2025.
- **_Mario at EXIST 2025: A Simple Gateway to Effective Multilingual Sexism Detection_.** https://ceur-ws.org/Vol-4038/paper_166.pdf
  Un Llama-3.1-8B fine-tuné en QLoRA, premier en 2025 : c'est le meilleur score de référence sur ce jeu de tweets.
- **A. Alajmi, G. Pergola, _Leveraging Model Confidence and Diversity: A Multi-Stage Framework for Sexism Detection_ (Warwick), EXIST 2025.** https://ceur-ws.org/Vol-4038/paper_137.pdf
  Les tweets sur lesquels le classifieur hésite sont envoyés à un LLM : 4e en 2025.
- **_When in Doubt, Consult: Expert Debate for Sexism Detection via Confidence-Based Routing_, ACL 2026.** https://arxiv.org/abs/2512.23732
  Ils font débattre plusieurs LLM sur les cas incertains, et donnent aussi des astuces d'entraînement sans LLM : pondération des classes, label smoothing, calibration.

### Augmentation de données

- **A. Karimi, L. Rossi, A. Prati, _AEDA: An Easier Data Augmentation Technique for Text Classification_, EMNLP Findings 2021.** https://arxiv.org/abs/2108.13230
  C'est le papier d'origine d'AEDA, qu'on a implémenté ; il montre qu'insérer de la ponctuation fait mieux qu'EDA.
- **J. Wei, K. Zou, _EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks_, EMNLP 2019.** https://arxiv.org/abs/1901.11196
  C'est la méthode de référence (synonymes, insertions, permutations, suppressions) ; on l'a écartée parce qu'elle peut changer le sens du tweet.

### Modèles utilisés

- **F. Barbieri, L. Espinosa Anke, J. Camacho-Collados, _XLM-T: Multilingual Language Models in Twitter for Sentiment Analysis and Beyond_, LREC 2022.** https://aclanthology.org/2022.lrec-1.27
  C'est le papier de `cardiffnlp/twitter-xlm-roberta-base` (https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base), le modèle de `Modelling.ipynb`.
- **D. Q. Nguyen, T. Vu, A. T. Nguyen, _BERTweet: A pre-trained language model for English Tweets_, EMNLP 2020 Demos.** https://aclanthology.org/2020.emnlp-demos.2
  C'est le papier de BERTweet (https://huggingface.co/vinai/bertweet-large), dont on a repris le format de normalisation des tweets (`@USER`, `HTTPURL`, emojis en texte).
- **`NLP-LTU/bertweet-large-sexism-detector`.** https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector
  C'est le BERTweet-large fine-tuné sur EDOS, utilisé comme point de départ du modèle « sexisme ».
- **MarianMT (Helsinki-NLP).** https://huggingface.co/Helsinki-NLP/opus-mt-es-en et https://huggingface.co/Helsinki-NLP/opus-mt-en-es
  Ce sont les modèles de traduction de l'augmentation testée dans `Modelling.ipynb`.
- **NLLB Team, _No Language Left Behind: Scaling Human-Centered Machine Translation_, 2022.** https://arxiv.org/abs/2207.04672
  C'est le papier de NLLB-200 (https://huggingface.co/facebook/nllb-200-3.3B), utilisé pour traduire l'espagnol en anglais dans `modelling2`.

### Choix du modèle de traduction

- **S. Kudugunta et al., _MADLAD-400: A Multilingual And Document-Level Large Audited Dataset_, 2023.** https://arxiv.org/abs/2309.04662
  Ce sont les modèles MADLAD-400, l'alternative à NLLB qu'on a envisagée.
- **_MT-LENS: An all-in-one Toolkit for Better Machine Translation Evaluation_, NAACL 2025 Demos.** https://aclanthology.org/2025.naacl-demo.6.pdf
  Il compare NLLB-3.3B et MADLAD-3B : MADLAD y est plus robuste aux fautes.
- **Évaluation de la traduction anglais / espagnol / catalan / espéranto, SIGUL 2026.** https://aclanthology.org/2026.sigul-1.31.pdf
  La famille NLLB y obtient les meilleurs scores sur ces paires de langues, ce qui a motivé le choix de NLLB.

### Outils

- **Extension Google Colab pour VS Code, guide d'utilisation.** https://github.com/googlecolab/colab-vscode/wiki/User-Guide
  Il explique comment envoyer les données au serveur Colab (« Upload to Colab ») et monter Google Drive.
- **Dépôt officiel de BERTweet.** https://github.com/VinAIResearch/BERTweet
  Il précise le prétraitement attendu par BERTweet : modèle sensible à la casse, `@USER`, `HTTPURL`, emojis convertis en texte.
