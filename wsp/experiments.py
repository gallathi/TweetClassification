"""Banc d'essai des éditions sur le vrai modèle (pas d'apprentissage, tout sous no_grad).

Chaque tentative :
  1. (optionnel) édition dans un MLP intermédiaire ;
  2. passe avant sur un sous-ensemble de calib -> statistiques de l'état final (d, m) ;
  3. (optionnel) édition des lignes Oui/Non de lm_head, biais choisi sur calib ;
  4. évaluation sur val (vraie passe avant du modèle édité) ;
  5. restauration des poids d'origine. Tout est consigné dans journal.jsonl.
"""
import json
import os
import sys
import time

import numpy as np
import torch

from common import (ROOT, answer_ids, fmt, load_model, load_splits, metrics, run,
                    text_model)

CACHE = ROOT / "wsp" / "cache"
JOURNAL = ROOT / "wsp" / "journal.jsonl"
N_CALIB = 2000


# ------------------------------------------------------------------ éditions élémentaires
def readout_edit(model, tok, H, y, s, a=1.0, bias=True, n_bias_dims=8):
    """Édite les lignes Oui/Non de lm_head (liées aux embeddings d'entrée de ces 2 tokens).

    1. Direction : lm_head[Oui] += Δ/2, lm_head[Non] -= Δ/2, Δ = a·d/||d||², où
       d = μ_sexiste - μ_non de l'état final normalisé (calib) : l'écart des moyennes de
       classe sur le score augmente de a.
    2. Biais : le score réel (après arrondi bf16) est recalculé sur calib, le décalage c qui
       maximise le F1 macro au seuil 0 est choisi, puis écrit sur les n_bias_dims coordonnées
       les plus stables de l'état final (h_j ≈ m_j, coefficient de variation ≈ 2 %).
    """
    oui, non = answer_ids(tok)
    W = model.lm_head.weight
    dev = W.device
    d = H[y == 1].mean(0) - H[y == 0].mean(0)
    if a:
        delta = torch.tensor(a * d / (d @ d), dtype=torch.float32, device=dev)
        W[oui] = (W[oui].float() + delta / 2).to(W.dtype)
        W[non] = (W[non].float() - delta / 2).to(W.dtype)
    r = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
    sc = H @ r  # score réel après édition (H = état final normalisé)
    c, dims = 0.0, []
    if bias:
        cs = np.quantile(sc, np.linspace(0.05, 0.95, 181))
        c = float(max(cs, key=lambda c: metrics(y, sc - c)["f1_macro"]))
        m, sd = H.mean(0), H.std(0)
        cand = np.where(np.abs(m) > 10)[0]
        dims = cand[np.argsort(sd[cand] / np.abs(m[cand]))][:n_bias_dims]
        for j in dims:  # h_j ≈ m_j  =>  Δs ≈ -c / n
            step = c / (len(dims) * m[j])
            W[oui, j] = (W[oui, j].float() - step / 2).to(W.dtype)
            W[non, j] = (W[non, j].float() + step / 2).to(W.dtype)
        r2 = (W[oui].float() - W[non].float()).cpu().numpy().astype(np.float64)
        realized = float(np.mean(H @ r2 - sc))
    return {"a": a, "c": c, "bias_dims": [int(j) for j in dims],
            "biais_realise": realized if bias else 0.0,
            "calib_f1m": float(metrics(y, sc - c)["f1_macro"])}


def sparse_key(A, y, n):
    """Clé creuse : différence de moyennes restreinte aux n neurones de plus grande taille
    d'effet |μ1 - μ0| / σ (statistique en forme fermée, aucun apprentissage)."""
    k = A[y == 1].mean(0) - A[y == 0].mean(0)
    eff = np.abs(k) / (A.std(0) + 1e-6)
    idx = np.argsort(-eff)[:n]
    ks = np.zeros_like(k)
    ks[idx] = k[idx]
    return ks, idx


def sparse_rome_edit(model, layer, ks, u, t, sep_key):
    """down_proj[:, idx] += γ û k̂ₛ[idx]ᵀ : seules les colonnes des n neurones changent.
    γ est fixé pour que l'écart des moyennes de classe écrit le long de û vaille t·||u||."""
    gamma = t * np.linalg.norm(u) / sep_key
    rome_edit(model, layer, gamma, ks, u)
    return gamma


def amplify_edit(model, layer, beta, dhat):
    """down_proj ← (I + β d̂d̂ᵀ) down_proj : amplifie la composante de la sortie du MLP sur d̂."""
    Wd = text_model(model)[0][layer].mlp.down_proj.weight
    dh = torch.tensor(dhat, dtype=torch.float32, device=Wd.device)
    Wf = Wd.float()
    Wd.copy_((Wf + beta * torch.outer(dh, dh @ Wf)).to(Wd.dtype))


def rome_edit(model, layer, gamma, k, u):
    """down_proj += γ û k̂ᵀ : si la clé « sexiste » k (diff. de moyennes des activations du MLP)
    est active, écrire û (direction sexiste du résidu) dans le flux résiduel."""
    Wd = text_model(model)[0][layer].mlp.down_proj.weight
    kt = torch.tensor(k / np.linalg.norm(k), dtype=torch.float32, device=Wd.device)
    ut = torch.tensor(u / np.linalg.norm(u), dtype=torch.float32, device=Wd.device)
    Wd.copy_((Wd.float() + gamma * torch.outer(ut, kt)).to(Wd.dtype))


# ------------------------------------------------------------------ banc d'essai
class Bench:
    def __init__(self):
        self.sp = load_splits()
        self.model, self.tok = load_model()
        rng = np.random.default_rng(0)
        self.cal = self.sp["calib"].iloc[rng.choice(len(self.sp["calib"]), N_CALIB, replace=False)]
        self.orig = {n: p.detach().clone() for n, p in self.model.named_parameters()
                     if "lm_head" in n or "down_proj" in n or "embed_tokens" in n}
        self.n_total = sum(p.numel() for p in self.model.parameters())

    def restore(self):
        params = dict(self.model.named_parameters())
        for n, w in self.orig.items():
            params[n].copy_(w)

    def n_modified(self):
        params = dict(self.model.named_parameters())
        return int(sum((params[n] != w).sum().item() for n, w in self.orig.items()))

    def attempt(self, name, hypothesis, mid=None, readout=True, a=1.0):
        t = time.time()
        info = {}
        if mid is not None:
            info["mid"] = mid(self.model)
        s_c, hs = run(self.model, self.tok, self.cal.tweet.tolist(), hidden=True)
        y_c = self.cal.y.values
        H = hs[:, -1].astype(np.float64)
        info["calib_auc_before_readout"] = metrics(y_c, s_c)["auc"]
        if readout:
            info["readout"] = readout_edit(self.model, self.tok, H, y_c, s_c.astype(np.float64), a=a)
        s_v = run(self.model, self.tok, self.sp["val"].tweet.tolist())
        np.save(CACHE / f"attempt_{name}_val.npy", s_v)
        mv = metrics(self.sp["val"].y.values, s_v)
        nmod = self.n_modified()
        rec = {"id": name, "hypothese": hypothesis, "edition": info, "poids_modifies": nmod,
               "pct_poids": 100 * nmod / self.n_total, "val": {k: v for k, v in mv.items()}}
        print(f"[{name}] {hypothesis}\n   {info}\n   poids modifiés {nmod} ({rec['pct_poids']:.5f} %)"
              f"\n   val {fmt(mv)}   ({time.time() - t:.0f}s)", flush=True)
        with open(JOURNAL, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        self.restore()
        assert self.n_modified() == 0
        return mv


if __name__ == "__main__":
    B = Bench()
    z = np.load(CACHE / "calib_base.npz")
    Hc, yc = z["hs"].astype(np.float32), z["y"]
    A13 = np.load(CACHE / "mlp_calib.npy", mmap_mode="r")
    which = sys.argv[1:] or ["E1", "E2"]

    def dres(idx):  # direction diff. de moyennes du résidu (indice hs) normalisée
        d = Hc[yc == 1, idx].mean(0) - Hc[yc == 0, idx].mean(0)
        return d / np.linalg.norm(d)

    if "E1" in which:
        B.attempt("E1", "Biais seul : le modèle sur-répond Oui (70 %) ; décaler le score", a=0.0)
    if "E2" in which:
        B.attempt("E2", "Réaligner la lecture Oui/Non sur la diff. de moyennes de l'état final + biais", a=1.0)
    if "E3" in which:
        for L, beta in [(13, 2.0), (13, 8.0), (20, 4.0)]:
            dh = dres(L + 1)  # sortie de la couche L = hs[L+1]
            B.attempt(f"E3-L{L}-b{beta}", f"Amplifier la direction sexiste dans la sortie du MLP {L} (β={beta}), puis E2",
                      mid=lambda m, L=L, beta=beta, dh=dh: (amplify_edit(m, L, beta, dh), {"layer": L, "beta": beta})[1])
    if "E4" in which:
        for L, gamma in [(13, 0.5), (13, 2.0), (22, 1.0)]:
            Al = np.asarray(A13[:, L], dtype=np.float32)
            k = Al[yc == 1].mean(0) - Al[yc == 0].mean(0)
            u = Hc[yc == 1, L + 1].mean(0) - Hc[yc == 0, L + 1].mean(0)
            B.attempt(f"E4-L{L}-g{gamma}", f"Rang 1 style ROME dans down_proj {L} : clé MLP sexiste -> direction sexiste (γ={gamma}), puis E2",
                      mid=lambda m, L=L, g=gamma, k=k, u=u: (rome_edit(m, L, g, k, u), {"layer": L, "gamma": g})[1])
    if "E2b" in which:
        # E2 refait pour avoir ses scores val (comparaison appariée)
        B.attempt("E2b", "E2 (lecture seule), rejoué pour le bootstrap apparié", a=1.0)
    if "E5" in which:
        L, n = 13, 16
        Al = np.asarray(A13[:, L], dtype=np.float32)
        ks, idx = sparse_key(Al, yc, n)
        u = Hc[yc == 1, L + 1].mean(0) - Hc[yc == 0, L + 1].mean(0)
        kn = ks / np.linalg.norm(ks)
        sep = float(kn @ (Al[yc == 1].mean(0) - Al[yc == 0].mean(0)))
        print("neurones choisis (couche 13):", idx.tolist(),
              " moyenne de la clé sur calib : %.3f, écart de classes : %.3f" % (float((Al @ kn).mean()), sep))
        for t in [float(x) for x in os.environ.get("E5_T", "0.5,1,2,4").split(",")]:
            B.attempt(f"E5-L{L}-n{n}-t{t}",
                      f"Rang 1 CREUX dans down_proj {L} : {n} neurones de plus forte taille d'effet -> direction sexiste (t={t}), puis E2",
                      mid=lambda m, t=t: {"layer": L, "n": n, "t": t, "neurones": idx.tolist(),
                                          "gamma": sparse_rome_edit(m, L, ks, u, t, sep)})
