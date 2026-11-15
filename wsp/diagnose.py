"""Diagnostic du modèle de base (sans aucun apprentissage).

1. Erreurs : biais de classe, langue, intention (DIRECT / REPORTED / JUDGEMENTAL), exemples.
2. Résidu couche par couche : différence de moyennes calculée sur « calib », évaluée sur « val »
   (AUC de la projection, séparation de Cohen), et alignement avec la direction de lecture
   W_U[Oui] - W_U[Non] (logit lens).
"""
import numpy as np
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from sklearn.metrics import roc_auc_score

from common import MODEL_ID, ROOT, answer_ids, fmt, load_splits, metrics

CACHE = ROOT / "wsp" / "cache"


def load(name):
    z = np.load(CACHE / f"{name}_base.npz")
    return z["s"], z["hs"].astype(np.float32), z["y"]


def rmsnorm(x, w, eps=1e-6):
    # Qwen3.5 : RMSNorm « (1 + w) » (poids centrés sur 0)
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * (1 + w)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    sp = load_splits()
    val, cal = sp["val"], sp["calib"]
    s_v, H_v, y_v = load("val")
    s_c, H_c, y_c = load("calib")

    # ------------------------------------------------------------------ 1. erreurs
    print("=== Base, val :", fmt(metrics(y_v, s_v)))
    print("=== Base, calib :", fmt(metrics(y_c, s_c)))
    # meilleur seuil a posteriori (diagnostic seulement : la calibration est-elle le problème ?)
    thr = np.quantile(s_c, np.linspace(0.05, 0.95, 91))
    best = max(thr, key=lambda t: metrics(y_c, s_c, t)["f1_macro"])
    print(f"Seuil optimal sur calib = {best:.2f} -> val :", fmt(metrics(y_v, s_v, best)))
    print("Score moyen : YES=%.2f NO=%.2f (std %.2f)" % (s_v[y_v == 1].mean(), s_v[y_v == 0].mean(), s_v.std()))

    p_v = (s_v > 0).astype(int)
    df = val.assign(s=s_v, pred=p_v)
    print("\nPar langue :")
    for lang, g in df.groupby("lang"):
        print(f"  {lang}: n={len(g)}", fmt(metrics(g.y.values, g.s.values)))
    print("\nRappel des YES par intention majoritaire (calib, plus grand) :")
    dfc = cal.assign(s=s_c, pred=(s_c > 0).astype(int))
    for it, g in dfc[dfc.y == 1].groupby("intent"):
        print(f"  {it:12s} n={len(g):4d} rappel={g.pred.mean():.2f} score moyen={g.s.mean():.2f}")
    print("  NO (pour comparaison) : %%Oui=%.2f score moyen=%.2f" % (dfc[dfc.y == 0].pred.mean(), dfc[dfc.y == 0].s.mean()))
    print("\nScore moyen selon le nombre de « oui » des annotateurs (calib) :")
    print(dfc.groupby("n_yes").s.agg(["count", "mean"]).round(2).T.to_string())

    pd_opts = dict(n=6, random_state=0)
    for title, sub in [("FAUX NÉGATIFS", df[(df.y == 1) & (df.pred == 0)]),
                       ("FAUX POSITIFS", df[(df.y == 0) & (df.pred == 1)])]:
        print(f"\n--- {title} ({len(sub)}) ---")
        for _, r in sub.sample(min(len(sub), 6), random_state=0).iterrows():
            print(f"[{r.lang} {r.n_yes}/6 {r.intent} s={r.s:+.2f}] {r.tweet[:220]!r}")

    # ------------------------------------------------------------------ 2. résidu
    path = hf_hub_download(MODEL_ID, "model.safetensors-00001-of-00001.safetensors")
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
        emb_key = [k for k in keys if k.endswith("language_model.embed_tokens.weight")][0]
        norm_key = [k for k in keys if k.endswith("language_model.norm.weight")][0]
        W = f.get_tensor(emb_key).float()
        g = f.get_tensor(norm_key).float().numpy()
    oui, non = answer_ids(tok)
    r = (W[oui] - W[non]).numpy()  # direction de lecture (lm_head lié aux embeddings)

    # contrôle : le dernier état caché donne-t-il bien le score ?
    last = H_v[:, -1]
    for name, x in [("hs[-1] brut", last), ("norm(hs[-1])", rmsnorm(last, g))]:
        print(f"contrôle {name}: corr avec s = {np.corrcoef(x @ r, s_v)[0, 1]:.3f}")

    print("\nCouche | AUC val (proj. diff. moyennes) | Cohen d | cos(d, lecture) | logit-lens Δs")
    L = H_c.shape[1]
    rows = []
    for l in range(L):
        mu1, mu0 = H_c[y_c == 1, l].mean(0), H_c[y_c == 0, l].mean(0)
        d = mu1 - mu0
        proj_v = H_v[:, l] @ d
        auc = roc_auc_score(y_v, proj_v)
        pc = H_c[:, l] @ d
        coh = (pc[y_c == 1].mean() - pc[y_c == 0].mean()) / pc.std()
        # alignement avec la lecture, dans l'espace après norm finale
        dn = rmsnorm(H_c[y_c == 1, l], g).mean(0) - rmsnorm(H_c[y_c == 0, l], g).mean(0)
        cos = dn @ r / (np.linalg.norm(dn) * np.linalg.norm(r))
        lens = rmsnorm(H_v[:, l], g) @ r
        lens_auc = roc_auc_score(y_v, lens)
        rows.append((l, auc, coh, cos, lens_auc))
        print(f"{l:6d} | {auc:.3f} | {coh:5.2f} | {cos:+.3f} | lens AUC {lens_auc:.3f}")
    np.save(CACHE / "layer_stats.npy", np.array(rows))
