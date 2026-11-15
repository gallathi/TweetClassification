"""Exactitude des réponses GÉNÉRÉES (pas seulement les logits) aux 30 questions factuelles,
pour base / A seul / B seul / A+B (poids mélangés depuis un modèle édité).
Usage : python wsp/factual_gen.py [chemin_modele_edite]"""
import re
import sys

import numpy as np
from safetensors.torch import load_file

from common import load_model
from side_effects import FACTS

SUFFIXES = [" Réponds Oui ou Non.", " Réponds Oui ou Non, puis justifie en une phrase."]


def answer(model, tok, q, suffix):
    p = tok.apply_chat_template([{"role": "user", "content": q + suffix}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    enc = tok(p, return_tensors="pt").to(model.device)
    g = tok.decode(model.generate(**enc, max_new_tokens=6, do_sample=False)[0, enc.input_ids.shape[1]:],
                   skip_special_tokens=True)
    m = re.search(r"\b(oui|non|yes|no|sí|si)\b", g.lower())
    return (1 if m.group(1) in ("oui", "yes", "sí", "si") else 0) if m else -1, g


if __name__ == "__main__":
    edited = sys.argv[1] if len(sys.argv) > 1 else "../edited_model"
    model, tok = load_model()
    E = load_file(f"{edited}/model.safetensors")
    P = dict(model.named_parameters())
    kA = [k for k in P if "layers.13.mlp.down_proj" in k][0]
    kB = [k for k in P if "embed_tokens" in k][0]
    orig = {k: P[k].detach().clone() for k in (kA, kB)}
    y = np.array([a for _, a in FACTS])
    for name, keys in [("base", []), ("A seul", [kA]), ("B seul", [kB]), ("A+B", [kA, kB])]:
        for k in (kA, kB):
            P[k].copy_(orig[k])
        for k in keys:
            P[k].copy_(E[k].to(P[k].device))
        for suf in SUFFIXES:
            res = [answer(model, tok, q, suf) for q, _ in FACTS]
            pred = np.array([r[0] for r in res])
            wrong = [q for (q, a), p in zip(FACTS, pred) if p != a]
            print(f"{name:7s} |{suf[:30]:30s}| exact={np.mean(pred == y):.2f} sans_reponse={np.mean(pred == -1):.2f} "
                  f"%Oui={np.mean(pred == 1):.2f}  erreurs: {'; '.join(w[:28] for w in wrong)}", flush=True)
