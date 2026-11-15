"""Outils communs : données, prompt, score Oui/Non, métriques.

Aucune fonction ici ne calcule de gradient : tout tourne sous torch.no_grad().
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

torch.set_grad_enabled(False)  # garde-fou global : aucun graphe de calcul

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
SEED = 42

QUESTION = "Ce texte est-il sexiste ? Réponds Oui ou Non."


# --------------------------------------------------------------------------- données
def _load(path: Path) -> pd.DataFrame:
    d = json.load(open(path, encoding="utf-8"))
    df = pd.DataFrame(list(d.values()))
    df["n_yes"] = df["labels_task1"].apply(lambda l: sum(x == "YES" for x in l))
    df["p_yes"] = df["n_yes"] / df["labels_task1"].str.len()
    df["y"] = np.select([df["p_yes"] > 0.5, df["p_yes"] < 0.5], [1, 0], default=-1)
    # intention majoritaire parmi les votes « oui » (tâche 2) : DIRECT / REPORTED / JUDGEMENTAL
    df["intent"] = df["labels_task2"].apply(
        lambda l: (Counter(x for x in l if x != "-").most_common(1) or [("-", 0)])[0][0])
    return df


def load_splits() -> dict[str, pd.DataFrame]:
    """Même protocole que rapport.md : val = 10 % du train (seed 42), test = dev.

    Les égalités 3-3 sont écartées. « calib » (le reste du train) ne sert qu'à calculer
    des statistiques d'activations (moyennes) : aucun entraînement.
    """
    train = _load(DATA / "EXIST2024_training.json")
    dev = _load(DATA / "EXIST2024_dev.json")
    is_val = np.random.default_rng(SEED).random(len(train)) < 0.10
    out = {"calib": train[~is_val], "val": train[is_val], "test": dev}
    return {k: v[v["y"] >= 0].reset_index(drop=True) for k, v in out.items()}


# --------------------------------------------------------------------------- modèle
def load_model(path: str = MODEL_ID, device: str | None = None, dtype=torch.bfloat16):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(device).eval()
    return model, tok


def text_model(model):
    """Renvoie (decoder_layers, final_norm, lm_head) du sous-modèle texte."""
    lm = model.model.language_model
    return lm.layers, lm.norm, model.lm_head


def build_prompt(tok, tweet: str) -> str:
    msgs = [{"role": "user", "content": f"Texte : « {tweet} »\n\n{QUESTION}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def answer_ids(tok) -> tuple[int, int]:
    oui = tok.encode("Oui", add_special_tokens=False)
    non = tok.encode("Non", add_special_tokens=False)
    assert len(oui) == 1 and len(non) == 1, (oui, non)
    return oui[0], non[0]


def run(model, tok, tweets, batch_size=16, hidden=False, max_len=384, other=None, prompt_fn=None):
    """Passe avant sur la dernière position.

    Renvoie le score s = logit(Oui) - logit(Non) et, si hidden=True, le résidu
    (sortie de chaque couche, à la dernière position) : tableau [N, n_layers+1, d].
    Si `other` est une liste, on y ajoute pour chaque exemple le plus grand logit hors Oui/Non.
    `prompt_fn` remplace le prompt de la tâche (par défaut build_prompt).
    """
    oui, non = answer_ids(tok)
    prompt_fn = prompt_fn or (lambda t: build_prompt(tok, t))
    dev = next(model.parameters()).device
    W = model.lm_head.weight
    readout = (W[oui].float() - W[non].float())  # logit(Oui) - logit(Non), calculé en fp32
    order = np.argsort([len(t) for t in tweets])  # regroupe les longueurs proches
    scores = np.zeros(len(tweets), dtype=np.float32)
    max_other = np.zeros(len(tweets), dtype=np.float32)
    hs = None
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([prompt_fn(tweets[i]) for i in idx], return_tensors="pt",
                  padding=True, truncation=True, max_length=max_len).to(dev)
        out = model.model(**enc, output_hidden_states=hidden, use_cache=False)
        h_last = out.last_hidden_state[:, -1]
        scores[idx] = (h_last.float() @ readout).cpu().numpy()
        if other is not None:
            lg = model.lm_head(h_last).float()
            lg[:, [oui, non]] = -float("inf")
            max_other[idx] = lg.max(-1).values.cpu().numpy()
        if hidden:
            h = torch.stack([x[:, -1] for x in out.hidden_states], 1).float().cpu().numpy()
            if hs is None:
                hs = np.zeros((len(tweets),) + h.shape[1:], dtype=np.float32)
            hs[idx] = h
    if other is not None:
        other.append(max_other)
    return (scores, hs) if hidden else scores


# --------------------------------------------------------------------------- métriques
def metrics(y, s, thr: float = 0.0) -> dict:
    p = (s > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "f1_macro": f1_score(y, p, average="macro"),
        "f1_yes": f1_score(y, p, pos_label=1, zero_division=0),
        "precision_yes": precision_score(y, p, zero_division=0),
        "recall_yes": recall_score(y, p, zero_division=0),
        "accuracy": (p == y).mean(),
        "pct_pred_yes": p.mean(),
        "auc": roc_auc_score(y, s),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def fmt(m: dict) -> str:
    return (f"F1macro={m['f1_macro']:.3f} F1yes={m['f1_yes']:.3f} P={m['precision_yes']:.3f} "
            f"R={m['recall_yes']:.3f} acc={m['accuracy']:.3f} %Oui={m['pct_pred_yes']:.2f} "
            f"AUC={m['auc']:.3f} cm={m['cm']}")


def count_params(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    txt = sum(p.numel() for n, p in model.named_parameters() if "visual" not in n)
    return {"total": tot, "text": txt}
