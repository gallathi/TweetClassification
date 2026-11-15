"""Capture (hooks) des activations intermédiaires des MLP, act(gate x) * up x, à la dernière
position, pour toutes les couches. Puis AUC sur val de la projection sur la différence de
moyennes (calculée sur calib). Aucun apprentissage.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import ROOT, build_prompt, load_model, load_splits, text_model

CACHE = ROOT / "wsp" / "cache"


def capture(model, tok, tweets, batch_size=16):
    layers, _, _ = text_model(model)
    buf = {}
    hooks = [l.mlp.down_proj.register_forward_pre_hook(
        lambda mod, inp, i=i: buf.__setitem__(i, inp[0][:, -1].float().cpu().numpy()))
        for i, l in enumerate(layers)]
    out = np.zeros((len(tweets), len(layers), layers[0].mlp.intermediate_size), dtype=np.float16)
    order = np.argsort([len(t) for t in tweets])
    dev = next(model.parameters()).device
    try:
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            enc = tok([build_prompt(tok, tweets[i]) for i in idx], return_tensors="pt",
                      padding=True).to(dev)
            model.model(**enc, use_cache=False)
            out[idx] = np.stack([buf[i] for i in range(len(layers))], 1)
    finally:
        for h in hooks:
            h.remove()
    return out


if __name__ == "__main__":
    sp = load_splits()
    model, tok = load_model()
    A = {k: capture(model, tok, sp[k].tweet.tolist()) for k in ["calib", "val"]}
    np.save(CACHE / "mlp_val.npy", A["val"])
    np.save(CACHE / "mlp_calib.npy", A["calib"])
    yc, yv = sp["calib"].y.values, sp["val"].y.values
    for l in range(A["val"].shape[1]):
        Ac, Av = A["calib"][:, l].astype(np.float32), A["val"][:, l].astype(np.float32)
        k = Ac[yc == 1].mean(0) - Ac[yc == 0].mean(0)
        print(f"couche {l:2d}  AUC val (clé MLP diff. moyennes) = {roc_auc_score(yv, Av @ k):.3f}")
