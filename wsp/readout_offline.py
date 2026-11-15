"""Éditions de la tête de sortie (lignes Oui/Non de lm_head), évaluées hors ligne.

hs[:, 24] = dernier état caché normalisé = exactement ce que lit lm_head (corrélation 1,000
avec le score) : s(h) = h · (W[Oui] - W[Non]). Une édition ΔW[Oui] = +Δ/2, ΔW[Non] = -Δ/2
donne s'(h) = s(h) + h · Δ, calculable sur le cache sans repasser dans le modèle.

Les hyperparamètres sont choisis sur « calib », la validation sert de juge.
"""
import numpy as np

from common import ROOT, fmt, metrics

CACHE = ROOT / "wsp" / "cache"


def load(name):
    z = np.load(CACHE / f"{name}_base.npz")
    return z["s"].astype(np.float64), z["hs"][:, 24].astype(np.float64), z["y"]


s_c, H_c, y_c = load("calib")
s_v, H_v, y_v = load("val")

mu1, mu0 = H_c[y_c == 1].mean(0), H_c[y_c == 0].mean(0)
d = mu1 - mu0
m = H_c.mean(0)
print("||d|| = %.3f  ||m|| = %.3f  cos(d, m) = %.3f" % (np.linalg.norm(d), np.linalg.norm(m),
                                                        d @ m / np.linalg.norm(d) / np.linalg.norm(m)))
proj_m = H_c @ m / (m @ m)
print("h·m/||m||² : moyenne %.3f, écart-type %.3f (≈ constante => biais propre)" % (proj_m.mean(), proj_m.std()))


def bias_dir(n_vec):
    """Vecteur b tel que h·b ≈ 1 pour tout h (biais), orthogonal à la direction ajoutée."""
    b = m / (m @ m)
    return b


def edited_scores(H, s, a, c, d_unit):
    # Δ = a * d̂_s + c * b, avec d̂_s normalisé pour que l'écart des moyennes de classe vaille a
    return s + a * (H @ d_unit) - c * (H @ bias_dir(None))


d_unit = d / (d @ d)  # (μ1 - μ0) · d_unit = 1
grid_a = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8]
print("\n   a  |  c* (calib) | calib F1m | val")
results = {}
for a in grid_a:
    sc = edited_scores(H_c, s_c, a, 0, d_unit)
    # biais choisi sur calib : on cherche c tel que s' - c soit le meilleur seuil 0
    cs = np.quantile(sc, np.linspace(0.05, 0.95, 181))
    c = max(cs, key=lambda c: metrics(y_c, sc - c)["f1_macro"])
    mc = metrics(y_c, sc - c)
    mv = metrics(y_v, edited_scores(H_v, s_v, a, 0, d_unit) - c)
    results[a] = (c, mc, mv)
    print(f"{a:5.2f} | {c:+.3f} | {mc['f1_macro']:.3f} | {fmt(mv)}")

# direction pure (sans la lecture d'origine) pour référence : jusqu'où la diff. de moyennes peut aller
pc = H_c @ d
cs = np.quantile(pc, np.linspace(0.05, 0.95, 181))
c = max(cs, key=lambda c: metrics(y_c, pc - c)["f1_macro"])
print("\nRéférence « diff. de moyennes seule » val :", fmt(metrics(y_v, H_v @ d - c)))
