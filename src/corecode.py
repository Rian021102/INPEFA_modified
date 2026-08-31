"""
INPEFA-DTW well correlation, built correctly this time:

  GR (or RT) -> L1 trend filtering cascade (9 lambda levels, user's
  original l1tf/l1tf_lm) -> AR32 Burg fit (spectrum._arburg2, order 32,
  as in the user's original PyNPEFAmain.inpefa()) -> PEFA (convolution
  with the AR coefficients) -> cumulative integration (reversed cumsum)
  -> normalized INPEFA curve (4 orders: long/mid/short/shorter-term)
  -> Dynamic Time Warping between wells' INPEFA curves, CONSTRAINED by
  a Sakoe-Chiba band (dtaidistance's `window` parameter), so the warping
  path can't drift arbitrarily far from the diagonal -- this prevents
  DTW from finding spuriously "good" alignments by warping one small
  feature onto a large stretch of the other curve, which is exactly the
  kind of degenerate match an unconstrained DTW can produce.

This corrects the earlier well_correlation_dtw.py / gr_dtw_correlation.py
scripts, which used only the bare L1 trend (no Burg/PEFA/cumsum) and
unconstrained DTW (no band) -- a materially different, simpler pipeline
than what was actually intended.
"""

import sys
sys.path.insert(0, "/home/claude/overlay")

import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from scipy import signal
from spectrum.burg import _arburg2
from cvxopt import matrix, solvers
from dtaidistance import dtw

from PyNPEFAmain import l1tf_lm, l1tf

solvers.options['show_progress'] = False

FILES = {
    "SRN-2": "/home/rian/python_project/myvenv/INPEFA_modified/data/SRN-2_Logs.las",
    "SRN-9": "/home/rian/python_project/myvenv/INPEFA_modified/data/SRN-9_Logs.las",
    "SRN-10": "/home/rian/python_project/myvenv/INPEFA_modified/data/SRN-10_Logs.las",
}
N_RESAMPLE = 300
SAKOE_CHIBA_FRACTION = 0.10  # band width as a fraction of sequence length


def inpefa_core(y):
    """Exact reproduction of the user's PyNPEFAmain.inpefa(), minus plotting."""
    y = np.asarray(y, dtype=float)
    lambdamax = l1tf_lm(y)

    z = {"0": matrix(y)}
    for i in range(1, 10):
        z[str(i)] = l1tf(z[str(i - 1)], 10 ** (-10 + i) * lambdamax)

    fy = {
        "1": z["0"] - (z["1"] + z["2"] + z["3"] + z["4"] + z["5"] + z["6"] + z["7"] + z["8"]) / 8.0,
        "2": z["0"] - (z["1"] + z["2"] + z["3"] + z["4"] + z["5"] + z["6"]) / 6.0,
        "3": z["0"] - (z["1"] + z["2"] + z["3"] + z["4"] + z["5"]) / 5.0,
        "4": z["0"] - (z["1"] + z["2"] + z["3"] + z["4"]) / 4.0,
    }

    ipfy = {"OG": y}
    for j in range(1, 5):
        fyj = np.asarray(fy[str(j)], dtype=float).ravel()
        bffy = _arburg2(fyj, 32)[0].real
        pffy = signal.convolve(fyj, bffy, mode="same")
        iipfy = np.cumsum(pffy[::-1])[::-1]
        ipfy[str(j)] = iipfy / max(abs(iipfy))
    return ipfy


def load_and_compute_inpefa(well, path, curve="GR"):
    las = lasio.read(path)
    df = las.df().reset_index()
    if curve == "RT":
        df = df.dropna(subset=["RT"])
        df = df[df["RT"] > 0].sort_values("DEPTH").reset_index(drop=True)
        vals = np.log10(df["RT"].to_numpy())
    else:
        df = df.dropna(subset=["GR"])
        df = df[(df["GR"] >= 0) & (df["GR"] <= 300)].sort_values("DEPTH").reset_index(drop=True)
        vals = df["GR"].to_numpy()

    depth = df["DEPTH"].to_numpy()
    print(f"  {well}: computing full INPEFA on {curve} ({len(vals)} points)...")
    ipfy = inpefa_core(vals)

    # resample the long-term order (order '1') onto common fractional depth
    frac_depth = (depth - depth.min()) / (depth.max() - depth.min())
    common_frac = np.linspace(0, 1, N_RESAMPLE)
    inpefa_order1 = np.interp(common_frac, frac_depth, ipfy["1"])

    if curve == "GR":
        inpefa_order1 = -inpefa_order1  # match RT's rise/fall sign convention

    return {"well": well, "curve": curve, "inpefa": inpefa_order1,
            "common_frac": common_frac, "thickness_ft": depth.max() - depth.min()}


def banded_dtw_distance(a, b, window_fraction=SAKOE_CHIBA_FRACTION):
    window = max(1, int(round(window_fraction * max(len(a), len(b)))))
    d, paths = dtw.warping_paths(a, b, window=window)
    path = dtw.best_path(paths)
    return d / len(path), path, window


def null_reference_distances(curve_type, n_shuffles=100, window_fraction=SAKOE_CHIBA_FRACTION, seed=0):
    """
    PROPER null for cumsum-integrated curves: run independent GAUSSIAN
    WHITE NOISE through the EXACT SAME pipeline (L1 trend -> Burg AR32
    -> PEFA -> cumsum -> normalize), rather than shuffling real INPEFA
    values directly.

    Why this matters: shuffling real INPEFA values destroys their
    natural autocorrelation/smoothness, making the null LESS smooth
    than genuine INPEFA curves -- this is not a fair reference for
    cumsum-integrated signals, which are well known (Granger-Newbold
    style "spurious regression" results) to show apparent similarity to
    ANY other integrated series purely from shared low-frequency
    "random walk" character, independent of whether the underlying
    data is truly related. The correct null is an INDEPENDENTLY
    integrated reference -- i.e., real noise pushed through the same
    integration machinery -- not a shuffled version of the real signal.
    """
    rng = np.random.default_rng(seed)
    null_dists = []
    n_points_guess = 5000
    for _ in range(n_shuffles):
        noise_a = rng.standard_normal(n_points_guess)
        noise_b = rng.standard_normal(n_points_guess)
        ipfy_a = inpefa_core(noise_a)["1"]
        ipfy_b = inpefa_core(noise_b)["1"]
        common_frac = np.linspace(0, 1, N_RESAMPLE)
        raw_frac = np.linspace(0, 1, n_points_guess)
        a = np.interp(common_frac, raw_frac, ipfy_a)
        b = np.interp(common_frac, raw_frac, ipfy_b)
        a_z = (a - a.mean()) / (a.std() + 1e-9)
        b_z = (b - b.mean()) / (b.std() + 1e-9)
        d, _, _ = banded_dtw_distance(a_z, b_z, window_fraction)
        null_dists.append(d)
    return np.array(null_dists)


def run_for_curve(curve):
    print(f"\n{'='*70}\nCURVE = {curve}\n{'='*70}")
    curves = {}
    for well, path in FILES.items():
        curves[well] = load_and_compute_inpefa(well, path, curve=curve)

    pairs = [("SRN-9", "SRN-10"), ("SRN-2", "SRN-9"), ("SRN-2", "SRN-10")]
    results = []
    for w1, w2 in pairs:
        a = curves[w1]["inpefa"]
        b = curves[w2]["inpefa"]
        a_z = (a - a.mean()) / (a.std() + 1e-9)
        b_z = (b - b.mean()) / (b.std() + 1e-9)
        d, path, window = banded_dtw_distance(a_z, b_z)
        results.append({"pair": f"{w1} vs {w2}", "curve": curve,
                         "dtw_distance": d, "sakoe_chiba_window": window})
        print(f"  {w1} vs {w2}: distance={d:.4f} (Sakoe-Chiba window={window} samples)")

    # need z-scored curves for the null too
    null_dists = null_reference_distances(curve, n_shuffles=40)
    print(f"  Null (banded, shuffled): mean={null_dists.mean():.4f}, std={null_dists.std():.4f}")

    for r in results:
        z = (r["dtw_distance"] - null_dists.mean()) / (null_dists.std() + 1e-9)
        r["z_vs_null"] = z
        print(f"    {r['pair']}: z-score vs null = {z:+.2f}")

    return curves, pd.DataFrame(results)


def main():
    all_results = []
    all_curves = {}
    for curve in ["RT", "GR"]:
        curves, results = run_for_curve(curve)
        all_results.append(results)
        all_curves[curve] = curves

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("inpefa_dtw_banded_results.csv", index=False)
    print(f"\nSaved -> inpefa_dtw_banded_results.csv")
    print("\n" + "="*70)
    print("Full comparison table:")
    print("="*70)
    print(combined.pivot(index="pair", columns="curve", values=["dtw_distance", "z_vs_null"]).round(3))

    plot_curves(all_curves)


def plot_curves(all_curves, out_path="inpefa_dtw_banded_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"SRN-2": "firebrick", "SRN-9": "steelblue", "SRN-10": "darkgreen"}
    for ax, curve in zip(axes, ["RT", "GR"]):
        for well, c in all_curves[curve].items():
            ax.plot(c["common_frac"], c["inpefa"], label=well, color=colors[well], lw=1.2)
        ax.set_xlabel("Fractional depth")
        ax.set_ylabel(f"INPEFA (order 1, long-term), from {curve}")
        ax.set_title(f"Full INPEFA curves -- {curve}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("True INPEFA (L1 trend + Burg AR32 + PEFA + cumsum) used for DTW correlation",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=140)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
