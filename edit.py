"""Applique l'édition « weight-space » à Qwen/Qwen3.5-0.8B pour la détection de sexisme.

Aucun entraînement : pas de gradient, pas d'optimiseur, pas de classifieur appris.
Tout tourne sous torch.no_grad(). Seules des passes avant, des moyennes, des différences de
moyennes, des écarts-types et des produits extérieurs (rang 1) sont utilisés.

Prompt fixe : « Ce texte est-il sexiste ? Réponds Oui ou Non. », décision = logit(Oui) > logit(Non).
Statistiques calculées sur « calib » = 90 % du train EXIST 2024 (égalités 3-3 écartées).

Édition A — rang 1 CREUX dans le MLP de la couche 13 (16 colonnes de down_proj) :
    a   = activations intermédiaires du MLP 13 (act(gate x)·up x) à la position de réponse
    k   = μ(a | sexiste) - μ(a | non sexiste), restreinte aux 16 neurones de plus grande
          taille d'effet |k_i| / σ_i, puis normalisée
    u   = μ(r | sexiste) - μ(r | non sexiste), r = résidu en sortie de la couche 13, normalisée
    down_proj ← down_proj + γ u kᵀ,  γ = t·||μ_r1 - μ_r0|| / (kᵀ(μ_a1 - μ_a0)),  t = 4
    => quand les « neurones du sexisme » s'activent, le MLP écrit la direction sexiste dans le
       flux résiduel avec t fois l'écart de classes déjà présent (t = 4).

Édition B — lignes Oui/Non de lm_head (liées aux embeddings d'entrée de ces 2 tokens) :
    h   = état final normalisé (après édition A), d = μ(h | sexiste) - μ(h | non sexiste)
    lm_head[Oui] += d/(2||d||²), lm_head[Non] -= d/(2||d||²)  (réaligne la lecture Oui/Non)
    biais c (meilleur F1 macro sur calib au seuil 0) écrit sur les 8 coordonnées les plus
    stables de h (h_j ≈ constante).

Le modèle original n'est jamais écrasé ; le modèle édité va dans ./edited_model/.

Usage : python edit.py [--out edited_model] [--layer 13] [--n-neurons 16] [--t 4] [--a 1]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "wsp"))
from common import (MODEL_ID, QUESTION, answer_ids, build_prompt, count_params,  # noqa: E402
                    fmt, load_model, load_splits, metrics, run, text_model)

torch.set_grad_enabled(False)


def capture_layer(model, tok, tweets, layer, batch_size=16):
    """Activations intermédiaires du MLP et sortie résiduelle de `layer`, dernière position."""
    L = text_model(model)[0][layer]
    buf = {}
    h1 = L.mlp.down_proj.register_forward_pre_hook(
        lambda m, inp: buf.__setitem__("a", inp[0][:, -1].float().cpu().numpy()))
    h2 = L.register_forward_hook(
        lambda m, inp, out: buf.__setitem__("r", (out[0] if isinstance(out, tuple) else out)[:, -1].float().cpu().numpy()))
    A = np.zeros((len(tweets), L.mlp.intermediate_size), np.float32)
    R = np.zeros((len(tweets), model.config.text_config.hidden_size), np.float32)
    order = np.argsort([len(t) for t in tweets])
    try:
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([build_prompt(tok, tweets[i]) for i in idx], return_tensors="pt",
                      padding=True).to(model.device)
            model.model(**enc, use_cache=False)
            A[idx], R[idx] = buf["a"], buf["r"]
    finally:
        h1.remove()
        h2.remove()
    return A, R


def edit_mlp(model, tok, df, layer=13, n_neurons=16, t=8.0):
    """Édition A : rang 1 creux dans down_proj[layer]."""
    y = df.y.values
    A, R = capture_layer(model, tok, df.tweet.tolist(), layer)
    k = A[y == 1].mean(0) - A[y == 0].mean(0)
    effect = np.abs(k) / (A.std(0) + 1e-6)
    idx = np.sort(np.argsort(-effect)[:n_neurons])
    ks = np.zeros_like(k)
    ks[idx] = k[idx]
    kn = ks / np.linalg.norm(ks)
    u = R[y == 1].mean(0) - R[y == 0].mean(0)
    sep_key = float(kn @ k)  # écart des moyennes de classe le long de la clé
    gamma = t * np.linalg.norm(u) / sep_key
    Wd = text_model(model)[0][layer].mlp.down_proj.weight
    ut = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=Wd.device)
    kt = torch.tensor(kn, dtype=torch.float32, device=Wd.device)
    Wd.copy_((Wd.float() + gamma * torch.outer(ut, kt)).to(Wd.dtype))
    return {"layer": layer, "neurones": idx.tolist(), "t": t, "gamma": float(gamma),
            "taille_effet_neurones": effect[idx].round(3).tolist(),
            "moyenne_cle_calib": float((A @ kn).mean()), "ecart_classes_cle": sep_key}


def final_states(model, tok, texts, prompt_fn=None):
    other = []
    s, hs = run(model, tok, texts, hidden=True, other=other, prompt_fn=prompt_fn)
    return s, hs[:, -1].astype(np.float64), other[0].astype(np.float64)


def edit_readout_specific(model, tok, df, a=1.0, cover=0.99):
    """Édition B (version spécifique à la tâche) : lignes Oui/Non de lm_head.

    m_t = état final moyen sur la tâche (calib), m_g = état final moyen sur des questions
    Oui/Non générales (jeu de référence). Dans le plan {m_t, m_g} on résout le système 2×2 :
        b_t·m_t = 1, b_t·m_g = 0     et     b_g·m_t = 0, b_g·m_g = 1.
    Puis, en une seule écriture (fp32 -> bf16) :
        Δ  = a·d/||d||² - (a·d/||d||² · m_g)·b_g    (direction sans effet moyen hors tâche)
        W[Oui] += Δ/2 - c·b_t/2 + δ·b_t ;  W[Non] += -Δ/2 + c·b_t/2 + δ·b_t
    c (biais) : meilleur F1 macro sur calib au seuil 0.  δ (format) : relève Oui et Non
    ensemble sur la tâche pour qu'ils restent l'argmax sur `cover` de calib ; ~0 hors tâche.
    """
    from generic_questions import REF, ref_prompt
    oui, non = answer_ids(tok)
    W = model.lm_head.weight
    w_oui = W[oui].float().cpu().numpy().astype(np.float64)
    w_non = W[non].float().cpu().numpy().astype(np.float64)
    r0 = w_oui - w_non

    s, H, mo = final_states(model, tok, df.tweet.tolist())
    _, Hg, _ = final_states(model, tok, REF, prompt_fn=lambda q: ref_prompt(tok, q))
    y = df.y.values
    m_t, m_g = H.mean(0), Hg.mean(0)
    M = np.stack([m_t, m_g], 1)
    B = M @ np.linalg.inv(M.T @ M)          # colonnes : b_t, b_g
    b_t, b_g = B[:, 0], B[:, 1]

    d = H[y == 1].mean(0) - H[y == 0].mean(0)
    u = a * d / (d @ d)
    delta = u - (u @ m_g) * b_g
    sc = H @ (r0 + delta)
    cs = np.quantile(sc, np.linspace(0.05, 0.95, 181))
    c = float(max(cs, key=lambda c: metrics(y, sc - c)["f1_macro"]))
    new_oui, new_non = w_oui + delta / 2 - c * b_t / 2, w_non - delta / 2 + c * b_t / 2
    # format : écart entre le meilleur autre token et max(Oui, Non), sur calib
    gap = mo - np.maximum(H @ new_oui, H @ new_non)
    dlt = float(max(0.0, np.quantile(gap, cover) + 0.1))
    new_oui, new_non = new_oui + dlt * b_t, new_non + dlt * b_t

    W[oui] = torch.tensor(new_oui, dtype=W.dtype, device=W.device)
    W[non] = torch.tensor(new_non, dtype=W.dtype, device=W.device)
    r2 = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
    wo2 = W[oui].float().cpu().numpy().astype(np.float64)
    wn2 = W[non].float().cpu().numpy().astype(np.float64)
    return {"mode": "specifique", "a": a, "bias_c": c, "format_delta": dlt,
            "cos_m_t_m_g": float(m_t @ m_g / np.linalg.norm(m_t) / np.linalg.norm(m_g)),
            "cos_d_lecture_avant": float(d @ r0 / np.linalg.norm(d) / np.linalg.norm(r0)),
            "decalage_moyen_score_ref": float((Hg @ (r2 - r0)).mean()),
            "decalage_moyen_score_tache": float((H @ (r2 - r0)).mean()),
            "argmax_oui_non_calib": float((np.maximum(H @ wo2, H @ wn2) > mo).mean()),
            "calib_avant_B": metrics(y, s), "calib_apres": metrics(y, H @ r2)}


def edit_readout(model, tok, df, a=1.0, n_bias_dims=8):
    """Édition B (version globale, tentatives E1-E5) : direction diff. de moyennes + biais écrit
    sur les 8 coordonnées les plus stables de h. Le biais agit sur TOUTE question Oui/Non."""
    s, hs = run(model, tok, df.tweet.tolist(), hidden=True)
    H, y = hs[:, -1].astype(np.float64), df.y.values
    oui, non = answer_ids(tok)
    W = model.lm_head.weight
    r0 = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
    d = H[y == 1].mean(0) - H[y == 0].mean(0)
    delta = torch.tensor(a * d / (d @ d), dtype=torch.float32, device=W.device)
    W[oui] = (W[oui].float() + delta / 2).to(W.dtype)
    W[non] = (W[non].float() - delta / 2).to(W.dtype)
    # biais choisi sur calib à partir du score réel (après arrondi bf16)
    r = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
    sc = H @ r
    cs = np.quantile(sc, np.linspace(0.05, 0.95, 181))
    c = float(max(cs, key=lambda c: metrics(y, sc - c)["f1_macro"]))
    m, sd = H.mean(0), H.std(0)
    cand = np.where(np.abs(m) > 10)[0]
    dims = cand[np.argsort(sd[cand] / np.abs(m[cand]))][:n_bias_dims]
    for j in dims:
        step = c / (len(dims) * m[j])
        W[oui, j] = (W[oui, j].float() - step / 2).to(W.dtype)
        W[non, j] = (W[non, j].float() + step / 2).to(W.dtype)
    r2 = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
    return {"a": a, "bias_c": c, "bias_dims": [int(j) for j in dims],
            "cos_d_lecture_avant": float(d @ r0 / np.linalg.norm(d) / np.linalg.norm(r0)),
            "calib_avant_B": metrics(y, s), "calib_apres": metrics(y, H @ r2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "edited_model"))
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--n-neurons", type=int, default=16)
    ap.add_argument("--t", type=float, default=4.0, help="0 = pas d'édition MLP")
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--bias-dims", type=int, default=8)
    ap.add_argument("--readout", choices=["specific", "global"], default="specific")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model(MODEL_ID)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()
            if "lm_head" in n or "embed_tokens" in n or f"layers.{args.layer}.mlp.down_proj" in n}
    calib = load_splits()["calib"]

    meta = {"model": MODEL_ID, "question": QUESTION, "calib_size": int(len(calib)),
            "tokens": dict(zip(["Oui", "Non"], answer_ids(tok)))}
    if args.t > 0:
        meta["edition_A_mlp"] = edit_mlp(model, tok, calib, args.layer, args.n_neurons, args.t)
        print("édition A (MLP) :", {k: v for k, v in meta["edition_A_mlp"].items() if k != "taille_effet_neurones"})
    if args.readout == "specific":
        meta["edition_B_lecture"] = edit_readout_specific(model, tok, calib, args.a)
    else:
        meta["edition_B_lecture"] = edit_readout(model, tok, calib, args.a, args.bias_dims)
    print("édition B :", {k: v for k, v in meta["edition_B_lecture"].items() if not k.startswith("calib")})
    print("calib après A, avant B :", fmt(meta["edition_B_lecture"]["calib_avant_B"]))
    print("calib final            :", fmt(meta["edition_B_lecture"]["calib_apres"]))

    params = dict(model.named_parameters())
    per_tensor = {n: int((params[n] != w).sum().item()) for n, w in orig.items()}
    n_mod = sum(per_tensor.values())
    n = count_params(model)
    meta.update({"poids_modifies_par_tenseur": per_tensor, "poids_modifies": n_mod,
                 "poids_total": n["total"], "poids_texte": n["text"],
                 "pct_modifies_total": 100 * n_mod / n["total"],
                 "pct_modifies_texte": 100 * n_mod / n["text"]})
    print(f"poids modifiés : {n_mod} / {n['total']} ({meta['pct_modifies_total']:.5f} %)  {per_tensor}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.jinja"]))
    for f in ["preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"]:
        if (snap / f).exists() and not (out / f).exists():
            shutil.copy(snap / f, out / f)
    json.dump(meta, open(out / "edit_meta.json", "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"modèle édité sauvegardé dans {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
