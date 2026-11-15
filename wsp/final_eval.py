"""Évaluation finale : UN SEUL passage sur le test (dev EXIST 2024, 934 tweets), base et édité.
Écrit wsp/final_results.json. À lancer une seule fois, après avoir figé l'édition.
"""
import gc
import json
import sys

import numpy as np
import torch

from common import MODEL_ID, ROOT, fmt, load_model, load_splits, metrics, run

OUT = ROOT / "wsp" / "final_results.json"

if __name__ == "__main__":
    assert not OUT.exists(), "le test a déjà été évalué : on ne le refait pas"
    edited = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "edited_model")
    sp = load_splits()
    res = {}
    for name, path in [("base", MODEL_ID), ("edite", edited)]:
        model, tok = load_model(path)
        res[name] = {}
        for split in ["val", "test"]:
            df = sp[split]
            s = run(model, tok, df.tweet.tolist())
            res[name][split] = {"all": metrics(df.y.values, s)}
            for lang in ["en", "es"]:
                mk = (df.lang == lang).values
                res[name][split][lang] = metrics(df.y.values[mk], s[mk])
            for it in ["DIRECT", "REPORTED", "JUDGEMENTAL"]:
                mk = (df.intent == it).values & (df.y.values == 1)
                res[name][split][f"rappel_{it}"] = float((s[mk] > 0).mean())
            res[name][split]["rappel_NO_faux_positifs"] = float((s[df.y.values == 0] > 0).mean())
            np.save(ROOT / "wsp" / "cache" / f"final_{name}_{split}.npy", s)
            print(f"{name:5s} {split:4s}", fmt(res[name][split]["all"]), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
