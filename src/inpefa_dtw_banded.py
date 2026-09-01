"""
INPEFA + Sakoe-Chiba banded DTW -- the computation core.

  log values -> L1 trend filtering cascade (9 lambda levels, user's
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

This module is pure computation: it takes arrays in and gives arrays and
distances back. It does NOT read LAS files, hold well paths, plot, or run
as a script -- that is mainpefa.py's job. mainpefa.py is the main code and
owns the one LAS loading sequence.
"""

import sys
sys.path.insert(0, "/home/claude/overlay")

import numpy as np
from scipy import signal
from spectrum.burg import _arburg2
from cvxopt import matrix, solvers
from dtaidistance import dtw

from PyNPEFAmain import l1tf_lm, l1tf

solvers.options['show_progress'] = False

N_RESAMPLE = 300
SAKOE_CHIBA_FRACTION = 0.10  # band width as a fraction of sequence length
INPEFA_ORDER = "1"           # long-term order used everywhere in this project


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


def orient_to_base_level(inpefa_curve, curve="GR"):
    """
    One sign convention for the whole project: a RISING curve means a
    base-level RISE. RT's INPEFA already has that sense; GR's is inverted
    (more shale = higher GR as base level rises), so GR is negated here --
    once, at the source, rather than separately per script.
    """
    return -inpefa_curve if curve == "GR" else inpefa_curve


def resample_to_fractional_depth(depth, curve, n_resample=N_RESAMPLE):
    """Put a curve on the common 0-1 fractional-depth grid used for DTW."""
    frac_depth = (depth - depth.min()) / (depth.max() - depth.min())
    common_frac = np.linspace(0, 1, n_resample)
    return common_frac, np.interp(common_frac, frac_depth, curve)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def banded_dtw_distance(a, b, window_fraction=SAKOE_CHIBA_FRACTION):
    """
    Sakoe-Chiba banded DTW, returning (distance, path, radius, cost_matrix).

    Two details that have to be right for this to be DTW in the textbook
    sense rather than just a number that behaves a bit like one:

    1. BAND RADIUS. The Sakoe-Chiba band of radius r is the constraint
       |i - j| <= r. dtaidistance's `window=w` argument actually admits
       |i - j| <= w - 1 (verified empirically against its cost matrix), so
       passing the radius straight through gives a band one cell narrower
       than advertised. We pass radius + 1 and report the true radius, so
       the band really is SAKOE_CHIBA_FRACTION of the sequence length.

    2. NORMALIZATION. dtaidistance returns d = sqrt(sum of SQUARED local
       costs along the optimal path). Dividing that by the path length K
       mixes units: sqrt(S)/K shrinks like 1/K, so pairs whose optimal path
       wanders (larger K) get an unearned advantage. The per-step quantity
       is the root-mean-square cost, sqrt(S/K) = d / sqrt(K), which is what
       we return -- a distance in the same units as the z-scored curves.

    Caveat worth stating: dtaidistance uses the symmetric1 step pattern
    (each of the three predecessors charged one local cost), for which no
    path normalization is exactly length-invariant -- only Sakoe & Chiba's
    symmetric2 pattern admits exact (N + M) normalization. Dividing by
    sqrt(K) is the standard practical choice; here every sequence is
    resampled to the same N_RESAMPLE anyway, so K varies only with how much
    the path warps, and distances stay comparable across pairs.
    """
    radius = max(1, int(round(window_fraction * max(len(a), len(b)))))
    d, paths = dtw.warping_paths(a, b, window=radius + 1)
    path = dtw.best_path(paths)
    return d / np.sqrt(len(path)), path, radius, paths


def random_walk_null_distances(n_shuffles=40, n_points=5000,
                               window_fraction=SAKOE_CHIBA_FRACTION, seed=0):
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
    for _ in range(n_shuffles):
        noise_a = rng.standard_normal(n_points)
        noise_b = rng.standard_normal(n_points)
        ipfy_a = inpefa_core(noise_a)[INPEFA_ORDER]
        ipfy_b = inpefa_core(noise_b)[INPEFA_ORDER]
        common_frac = np.linspace(0, 1, N_RESAMPLE)
        raw_frac = np.linspace(0, 1, n_points)
        a = np.interp(common_frac, raw_frac, ipfy_a)
        b = np.interp(common_frac, raw_frac, ipfy_b)
        d, _, _, _ = banded_dtw_distance(zscore(a), zscore(b), window_fraction)
        null_dists.append(d)
    return np.array(null_dists)
