"""Garde-fous : le modèle édité génère-t-il toujours un texte cohérent, et ne répond-il pas
« Oui » à tout ? Compare base et édité (chargés l'un après l'autre, 4 Go de VRAM).

Usage : python wsp/guardrails.py [chemin_modele_edite]
"""
import gc
import json
import sys

import numpy as np
import torch

from common import MODEL_ID, ROOT, answer_ids, build_prompt, load_model, load_splits, run

GENERAL = [
    "Explique en trois phrases comment fonctionne la photosynthèse.",
    "Write a short poem about the sea.",
    "¿Cuál es la capital de Argentina y qué se puede visitar allí?",
    "Donne une recette simple de crêpes.",
    "What is 17 * 23? Show your steps.",
]
OUI_NON = [  # la réponse commence naturellement par Oui/Non -> teste l'embedding d'entrée édité
    "Paris est-elle la capitale de la France ? Réponds Oui ou Non, puis justifie en une phrase.",
    "Le soleil tourne-t-il autour de la Terre ? Réponds Oui ou Non, puis justifie en une phrase.",
    "L'eau bout-elle à 100 °C au niveau de la mer ? Réponds Oui ou Non, puis explique pourquoi.",
]
NEUTRES = [  # aucun n'est sexiste
    "Il fait très beau aujourd'hui, on va à la plage avec les enfants.",
    "Je viens de finir mon match de foot, on a gagné 3-1 !",
    "Women deserve equal pay for equal work. #EqualPay",
    "Las mujeres obtuvieron el derecho al voto en España en 1931.",
    "My sister just got promoted to lead engineer, so proud of her!",
    "El nuevo restaurante del centro tiene una paella buenísima.",
    "Reminder: the meeting is moved to 3pm tomorrow.",
    "Feliz día a todas las madres que luchan cada día.",
]
SEXISTES = [
    "Women should stay in the kitchen and leave politics to men.",
    "Las mujeres no saben conducir, es un hecho.",
    "She only got the job because she's pretty, obviously.",
    "Calladita estás más guapa.",
]


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def generate(model, tok, prompts, n=80):
    outs = []
    for p in prompts:
        enc = tok(chat(tok, p), return_tensors="pt").to(model.device)
        g = model.generate(**enc, max_new_tokens=n, do_sample=False)
        outs.append(tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True))
    return outs


def next_token_logprobs(model, tok, texts):
    res = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        res.append(torch.log_softmax(model(**enc).logits[0].float(), -1).cpu())
    return res


def argmax_answers(model, tok, tweets):
    oui, non = answer_ids(tok)
    c = {"Oui": 0, "Non": 0, "autre": 0}
    for t in tweets:
        enc = tok(build_prompt(tok, t), return_tensors="pt").to(model.device)
        top = model(**enc, logits_to_keep=1).logits[0, -1].argmax().item()
        c["Oui" if top == oui else "Non" if top == non else "autre"] += 1
    return c


def report(path):
    model, tok = load_model(path)
    val = load_splits()["val"].sample(100, random_state=0)
    corpus = [  # texte général pour la KL du token suivant
        "La Révolution française débute en 1789 avec la convocation des États généraux.",
        "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
        "El fútbol es el deporte más popular de Sudamérica, especialmente en Argentina y Brasil.",
        "Oui, je pense que c'est une bonne idée. Non, je ne viendrai pas demain.",
    ]
    out = {
        "general": generate(model, tok, GENERAL),
        "oui_non": generate(model, tok, OUI_NON, n=50),
        "neutres_s": run(model, tok, NEUTRES).round(3).tolist(),
        "sexistes_s": run(model, tok, SEXISTES).round(3).tolist(),
        "argmax_val100": argmax_answers(model, tok, val.tweet.tolist()),
        "logprobs": next_token_logprobs(model, tok, corpus),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    edited = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "edited_model")
    B, E = report(MODEL_ID), report(edited)
    kl = [float((b.exp() * (b - e)).sum(-1).mean()) for b, e in zip(B["logprobs"], E["logprobs"])]
    top1 = [float((b.argmax(-1) == e.argmax(-1)).float().mean()) for b, e in zip(B["logprobs"], E["logprobs"])]
    print("KL(base || édité) moyenne par token, texte général :", np.round(kl, 5))
    print("accord top-1 du token suivant :", np.round(top1, 3))
    for k, lab in [("neutres_s", "phrases NON sexistes"), ("sexistes_s", "phrases sexistes")]:
        print(f"\n{lab} : score s (>0 = Oui)\n  base  {B[k]}\n  édité {E[k]}")
    print("\nRéponse réelle (argmax sur tout le vocabulaire), 100 tweets val :",
          "base", B["argmax_val100"], "| édité", E["argmax_val100"])
    for k in ["general", "oui_non"]:
        for i, (b, e) in enumerate(zip(B[k], E[k])):
            print(f"\n### {k}[{i}]\n--- base : {b!r}\n--- édité : {e!r}")
    res = {k: {"base": B[k], "edite": E[k]} for k in ["general", "oui_non", "neutres_s", "sexistes_s", "argmax_val100"]}
    res["kl"], res["top1"] = kl, top1
    json.dump(res, open(ROOT / "wsp" / "guardrails.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
