"""Effets de bord de l'édition hors de la tâche :
  1. questions factuelles Oui/Non (15 Oui, 15 Non) : exactitude via logit(Oui) - logit(Non) ;
  2. tâche sexisme : quels tokens l'argmax produit-il quand ce n'est ni Oui ni Non ?
Usage : python wsp/side_effects.py chemin_modele [chemin_modele ...]
"""
import gc
import sys
from collections import Counter

import numpy as np
import torch

from common import answer_ids, build_prompt, load_model, load_splits

FACTS = [  # (question, réponse attendue : 1 = Oui)
    ("Paris est-elle la capitale de la France ?", 1), ("L'eau bout-elle à 100 °C au niveau de la mer ?", 1),
    ("La Terre tourne-t-elle autour du Soleil ?", 1), ("Le chat est-il un mammifère ?", 1),
    ("2 + 2 font-ils 4 ?", 1), ("Le Japon est-il une île ?", 1), ("L'oxygène est-il un gaz à température ambiante ?", 1),
    ("Madrid est-elle en Espagne ?", 1), ("Une semaine compte-t-elle sept jours ?", 1),
    ("Le français est-il une langue romane ?", 1), ("La baleine vit-elle dans l'océan ?", 1),
    ("Un triangle a-t-il trois côtés ?", 1), ("La neige est-elle froide ?", 1),
    ("Shakespeare était-il anglais ?", 1), ("Le cœur pompe-t-il le sang ?", 1),
    ("Le Soleil tourne-t-il autour de la Terre ?", 0), ("Berlin est-elle la capitale de l'Italie ?", 0),
    ("Les poissons respirent-ils avec des poumons ?", 0), ("3 + 3 font-ils 7 ?", 0),
    ("La Lune est-elle plus grande que la Terre ?", 0), ("Les pingouins vivent-ils au pôle Nord ?", 0),
    ("Le fer est-il un gaz à température ambiante ?", 0), ("Un carré a-t-il cinq côtés ?", 0),
    ("L'Australie est-elle en Europe ?", 0), ("Les araignées ont-elles six pattes ?", 0),
    ("Le Nil est-il en Amérique du Sud ?", 0), ("L'eau gèle-t-elle à 50 °C ?", 0),
    ("Victor Hugo était-il allemand ?", 0), ("Une année compte-t-elle 400 jours ?", 0),
    ("Le sel est-il sucré ?", 0),
]


def facts(model, tok):
    oui, non = answer_ids(tok)
    s = []
    for q, _ in FACTS:
        p = tok.apply_chat_template([{"role": "user", "content": q + " Réponds Oui ou Non."}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        lg = model(**tok(p, return_tensors="pt").to(model.device), logits_to_keep=1).logits[0, -1].float()
        s.append((lg[oui] - lg[non]).item())
    s, y = np.array(s), np.array([a for _, a in FACTS])
    return s, ((s > 0) == y).mean()


def other_tokens(model, tok, tweets):
    oui, non = answer_ids(tok)
    c = Counter()
    for t in tweets:
        lg = model(**tok(build_prompt(tok, t), return_tensors="pt").to(model.device), logits_to_keep=1).logits[0, -1]
        top = lg.argmax().item()
        c[tok.decode(top) if top not in (oui, non) else ("Oui" if top == oui else "Non")] += 1
    return c


if __name__ == "__main__":
    tweets = load_splits()["val"].sample(100, random_state=0).tweet.tolist()
    for path in sys.argv[1:]:
        model, tok = load_model(path)
        s, acc = facts(model, tok)
        y = np.array([a for _, a in FACTS])
        print(f"\n== {path}\n  questions factuelles : exactitude {acc:.2f} | score moyen Oui-attendu {s[y == 1].mean():+.2f}"
              f" / Non-attendu {s[y == 0].mean():+.2f} | erreurs : "
              + "; ".join(q for (q, a), v in zip(FACTS, s) if (v > 0) != a))
        print("  argmax tâche sexisme (100 val) :", dict(other_tokens(model, tok, tweets).most_common(8)))
        del model
        gc.collect()
        torch.cuda.empty_cache()
