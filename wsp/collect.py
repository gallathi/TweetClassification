"""Passe avant du modèle de base sur un split : scores Oui/Non + résidu par couche.

Usage : python wsp/collect.py val calib      (jamais « test » avant l'évaluation finale)
"""
import sys
import time

import numpy as np

from common import ROOT, fmt, load_model, load_splits, metrics, run

CACHE = ROOT / "wsp" / "cache"
CACHE.mkdir(exist_ok=True)

if __name__ == "__main__":
    splits = load_splits()
    for k, v in splits.items():
        print(k, len(v), "YES=%.2f" % v.y.mean())
    model, tok = load_model()
    for name in sys.argv[1:]:
        assert name != "test", "le test ne se regarde qu'à la fin"
        df = splits[name]
        t = time.time()
        s, hs = run(model, tok, df.tweet.tolist(), hidden=True)
        np.savez(CACHE / f"{name}_base.npz", s=s, hs=hs.astype(np.float16), y=df.y.values)
        print(f"{name}: {time.time() - t:.0f}s  hs={hs.shape}")
        print("  ", fmt(metrics(df.y.values, s)))
