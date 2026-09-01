"""
Angle-based BELL/FUNNEL/BLOCKY shape typing using the FULL INPEFA
pipeline (L1 trend cascade + Burg AR32 + PEFA + cumsum -- inpefa_core,
the user's original algorithm) on GR ONLY.

Segmentation is at local extrema of the full INPEFA curve (order 1,
long-term) since -- unlike the bare L1 trend -- the full INPEFA output
is not piecewise-linear (verified: 100% of points have non-negligible
second difference), so there's no sparse "true kink" structure to
detect directly; extrema (peaks/troughs) are the natural segmentation
points instead, consistent with Yuan et al.'s own turning-point
definition.

Angle geometry: fixed 100 ft depth reference (not global fractional
depth -- that normalization collapsed all angles toward +/-90 degrees
in an earlier version; fixed by scaling depth against a constant
physical reference length instead), z-scored INPEFA curve, arctan2.

Sign convention: for GR, a base-level RISE (positive INPEFA trend)
means the curve INCREASES upward (more shale as base level rises) --
i.e. trend DECREASES upward = FALL for GR is the same physical sense
as RT's rise/fall, so the GR INPEFA curve is negated to keep
BELL = fining-up = rise and FUNNEL = coarsening-up = fall consistent
with the rest of this project. That negation is applied once, by
inpefa_dtw_banded.orient_to_base_level(), which is the single place
this project decides curve polarity (it used to be applied here a
second time, to the angle instead of the curve).
"""

"""
Angle-based GR shape typing following the Emery & Myers (1996) log-shape
taxonomy (Cylindrical / Funnel / Bell / Symmetrical / Serrated), using
the SPECIFIC slope-angle thresholds shown in the Pertamina Hulu
Kalimantan Timur "Facies Prediction in Deltaic System" workflow slide.

--------------------------------------------------------------------------
Citation
--------------------------------------------------------------------------
Shape taxonomy (Cylindrical/Funnel/Bell/Symmetrical/Serrated and their
aggrading/prograding/retrograding depositional interpretation):
    Emery, D. & Myers, K.J. (1996). Sequence Stratigraphy.
    Oxford: Blackwell Science, 297 p.
    DOI: 10.1002/9781444313710
    (now hosted on Wiley Online Library, having acquired Blackwell's
    catalog; ISBN 0-632-03706-7)

This taxonomy itself descends from the earlier bell/funnel/cylinder
scheme developed by Shell (Serra & Sulpice 1975) and popularised by
Selley (1978) and Rider (1990) -- Emery & Myers is the most commonly
cited modern reference and the one named on the source slide.

IMPORTANT CAVEAT: Emery & Myers (1996) is a textbook describing the
QUALITATIVE shape catalog; it is not itself the source of the specific
NUMERIC slope-angle cutoffs (32 degrees, 72 degrees) used below. Those
cutoffs come from the Pertamina Hulu Kalimantan Timur workflow slide
provided by the user -- an operationalization of the qualitative
taxonomy into an automated slope-filter rule, not a value quoted
verbatim from the original academic source. Flagged explicitly so the
numeric thresholds aren't misattributed.

--------------------------------------------------------------------------
Angle thresholds (from the slide's Filter 2 protractor diagram)
--------------------------------------------------------------------------
The angle is measured along the segment in the direction of INCREASING
depth (top -> base), on the base-level-oriented INPEFA curve, so:

 -90 deg <= angle < -32 deg  -> BELL           (fining-upward, retrograding)
  32 deg < angle <= 90 deg   -> FUNNEL         (coarsening-upward, prograding)
 -32 deg <= angle <= 32 deg  -> CYLINDRICAL    (uniform/blocky, aggrading)

(the magnitudes are the slide's; the sign here just follows the
top-to-base measurement direction the code uses -- verified against the
raw GR: BELL segments do have higher GR at their top, FUNNEL lower)

SYMMETRICAL: an adjacent BELL-then-FUNNEL or FUNNEL-then-BELL couplet
(a complete rise-then-fall parasequence) -- this is the same mechanism
as the "symmetric_couplet" flag used throughout this project, now
explicitly relabeled to match Emery & Myers' terminology.

SERRATED: repeated, thin, rapidly-alternating BELL/FUNNEL segments --
NOT given an explicit quantitative rule on the source slide, so this is
my own operationalization (flagged as such): 3 or more consecutive
segments that alternate BELL/FUNNEL AND are all thinner than the well's
own median segment thickness are tagged SERRATED, on top of their
individual BELL/FUNNEL label (a compound pattern flag, not a
replacement category -- consistent with how the classic literature
describes "serrated" as a texture superimposed on an underlying
funnel/bell/cylindrical trend, e.g. "funnel to serrated").

Same pipeline otherwise as angle_shape_fullINPEFA_GR.py: full INPEFA
(L1 trend cascade + Burg AR32 + PEFA + cumsum) on GR, segmented at
local extrema, angle via fixed-100ft-reference / z-scored-curve
geometry, and Sakoe-Chiba banded DTW well correlation (unchanged from
before -- already satisfies the "include Sakoe-Chiba band" request).
"""

import sys
sys.path.insert(0, "/home/claude/overlay")

import re
from pathlib import Path

import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.signal import argrelextrema

# inpefa_dtw_banded is the computation core only -- the INPEFA pipeline, the
# sign convention, fractional-depth resampling, the banded DTW and its
# random-walk null. Everything else (well paths, LAS loading, the run itself)
# lives here, in the main script.
from inpefa_dtw_banded import (
    INPEFA_ORDER,
    SAKOE_CHIBA_FRACTION,
    banded_dtw_distance,
    inpefa_core,
    orient_to_base_level,
    random_walk_null_distances,
    resample_to_fractional_depth,
    zscore,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_DIR = Path(__file__).resolve().parent.parent / "csv_out"
IMAGE_DIR = Path(__file__).resolve().parent.parent / "image_output"
SMOOTH_WIN = 201
DEPTH_REF_FT = 100.0

# The gamma-ray log is not always called GR -- some of these LAS files write
# it as GAMMA. Matched case-insensitively, first alias present wins, and the
# column is renamed to GR so the rest of the script has one name to work with.
GR_ALIASES = ("GR", "GAMMA")

# Colors are handed out per well in discovery order, so any number of wells
# gets a stable, distinct color across every plot in this script.
WELL_PALETTE = ["firebrick", "steelblue", "darkgreen", "darkorange", "purple",
                "teal", "saddlebrown", "magenta", "olive", "navy"]

# --- Emery & Myers / Pertamina slide thresholds ---
BELL_FUNNEL_LOWER_DEG = 32.0   # |angle| below this -> CYLINDRICAL
BELL_FUNNEL_UPPER_DEG = 72.0   # kept for reference/plotting; angles beyond
                                # this are still BELL/FUNNEL (even more
                                # extreme fining/coarsening), not a separate
                                # category -- the slide's pie chart only
                                # explicitly colors the 32-72 deg wedge, but
                                # there's no principled reason a steeper
                                # angle should stop being BELL/FUNNEL

SHAPE_COLORS = {"BELL": "#2ca02c", "FUNNEL": "#1f77b4", "CYLINDRICAL": "#7f7f7f"}


def _natural_key(name):
    """Sort WELL-2 before WELL-10 (plain string sort would not)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def discover_wells(data_dir=None):
    """
    Every .las in the data folder is a well. The well name is the file stem
    with a trailing "_Logs"/"_logs" dropped, so ABC-1_Logs.las -> "ABC-1".
    Returns {well: path}, naturally sorted.

    data_dir defaults to DATA_DIR, but is read at CALL time (not bound as a
    default argument), so overriding DATA_DIR or passing a folder in works.
    """
    data_dir = Path(data_dir if data_dir is not None else DATA_DIR)
    paths = [p for p in data_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".las"]
    if not paths:
        raise FileNotFoundError(f"No .las files found in {data_dir}")

    wells = {}
    for path in sorted(paths, key=lambda p: _natural_key(p.stem)):
        well = re.sub(r"_logs$", "", path.stem, flags=re.IGNORECASE)
        if well in wells:
            raise ValueError(f"Two LAS files map to the same well name {well!r}: "
                             f"{wells[well].name} and {path.name}")
        wells[well] = path
    return wells


def well_colors(wells):
    """One color per well, cycling the palette if there are more wells than colors."""
    return {w: WELL_PALETTE[i % len(WELL_PALETTE)] for i, w in enumerate(wells)}


def resolve_gr_column(df, source=""):
    """
    Find the gamma-ray curve whatever this particular LAS calls it: GR,
    GAMMA, any capitalization. Returns the actual column name.
    """
    lookup = {str(c).upper(): c for c in df.columns}
    for alias in GR_ALIASES:
        if alias in lookup:
            return lookup[alias]
    raise KeyError(f"No gamma-ray curve in {source or 'LAS file'}: looked for "
                   f"{'/'.join(GR_ALIASES)} (case-insensitive), "
                   f"found {list(df.columns)}")


def load_well(path):
    """
    THE LAS loading sequence for this project: find the gamma-ray curve under
    whichever mnemonic this file uses and normalize it to GR, drop nulls, keep
    it in its valid range, sort by depth, reset the index.
    """
    las = lasio.read(path)
    df = las.df().reset_index()

    gr_col = resolve_gr_column(df, Path(path).name)
    if gr_col != "GR":
        print(f"  gamma-ray curve is {gr_col!r} -- read as GR")
        df = df.rename(columns={gr_col: "GR"})

    df = df.dropna(subset=["GR"])
    df = df[(df["GR"] >= 0) & (df["GR"] <= 300)]
    return df.sort_values("DEPTH").reset_index(drop=True)


def smooth(y, window):
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    padded = np.pad(y, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(y)]


def find_extrema(curve, window=SMOOTH_WIN):
    s = smooth(curve, window)
    max_idx = argrelextrema(s, np.greater, order=window)[0]
    min_idx = argrelextrema(s, np.less, order=window)[0]
    return np.sort(np.concatenate([max_idx, min_idx]))


def segment_by_extrema(df, curve, ext_idx, depth_ref_ft=DEPTH_REF_FT):
    depth = df["DEPTH"].to_numpy()
    gr = df["GR"].to_numpy()
    curve_z = zscore(curve)

    boundaries = np.unique(np.concatenate([[0], ext_idx, [len(depth) - 1]]))

    segments = []
    for k in range(len(boundaries) - 1):
        i0, i1 = boundaries[k], boundaries[k + 1]
        if i1 <= i0:
            continue
        thickness = depth[i1] - depth[i0]
        if thickness <= 0:
            continue
        d_curve_z = curve_z[i1] - curve_z[i0]
        d_depth_ref = thickness / depth_ref_ft
        angle_deg = np.degrees(np.arctan2(d_curve_z, d_depth_ref))

        seg_gr = gr[i0:i1 + 1]
        segments.append({
            "top_depth": depth[i0], "base_depth": depth[i1],
            "thickness_ft": thickness, "angle_deg": angle_deg,
            "gr_mean": seg_gr.mean(), "gr_std": seg_gr.std(),
        })
    return pd.DataFrame(segments)


def classify_by_angle(seg_df):
    """
    BELL: -90 deg <= angle < -32 deg (fining-upward, retrograding)
    FUNNEL: 32 deg < angle <= 90 deg (coarsening-upward, prograding)
    CYLINDRICAL: -32 deg <= angle <= 32 deg (uniform/blocky, aggrading)
    Thresholds per the Pertamina Hulu Kalimantan Timur workflow slide;
    signs follow the top-to-base measurement direction (see module
    docstring).
    """
    shape = []
    for _, row in seg_df.iterrows():
        a = row["angle_deg"]
        if abs(a) <= BELL_FUNNEL_LOWER_DEG:
            shape.append("CYLINDRICAL")
        elif a < 0:
            shape.append("BELL")
        else:
            shape.append("FUNNEL")
    seg_df["shape"] = shape

    # SYMMETRICAL: adjacent BELL<->FUNNEL couplet (complete parasequence)
    is_symmetrical = [False] * len(seg_df)
    shapes = seg_df["shape"].to_list()
    for i in range(len(shapes) - 1):
        if {shapes[i], shapes[i + 1]} == {"BELL", "FUNNEL"}:
            is_symmetrical[i] = True
            is_symmetrical[i + 1] = True
    seg_df["symmetrical"] = is_symmetrical

    # SERRATED (my own operationalization -- not on the source slide, see
    # module docstring): 3+ consecutive BELL/FUNNEL-alternating segments,
    # all thinner than the well's own median segment thickness.
    median_thickness = seg_df["thickness_ft"].median()
    is_serrated = [False] * len(seg_df)
    i = 0
    while i < len(shapes) - 2:
        run = [i]
        j = i
        while (j + 1 < len(shapes)
               and shapes[j] in ("BELL", "FUNNEL")
               and shapes[j + 1] in ("BELL", "FUNNEL")
               and shapes[j] != shapes[j + 1]
               and seg_df.iloc[j]["thickness_ft"] <= median_thickness):
            j += 1
            run.append(j)
        if len(run) >= 3:
            for k in run:
                is_serrated[k] = True
            i = j + 1
        else:
            i += 1
    seg_df["serrated"] = is_serrated

    return seg_df


# --------------------------------------------------------------------------
# DTW well-to-well correlation, on the SAME full-INPEFA(GR) curves used for
# shape typing above -- Sakoe-Chiba banded, with a proper random-walk null
# (not a naive shuffled null -- shown earlier to be a poor reference for
# cumsum-integrated curves, since it destroys the natural smoothness that
# integration produces and biases the null toward "too easy to beat").
# --------------------------------------------------------------------------

def run_dtw_correlation(well_data):
    print(f"\n{'='*70}\nDTW well-to-well correlation (full INPEFA, GR, "
          f"Sakoe-Chiba band={SAKOE_CHIBA_FRACTION:.0%})\n{'='*70}")

    resampled = {}
    for well, data in well_data.items():
        _, r = resample_to_fractional_depth(data["df"]["DEPTH"].to_numpy(),
                                            data["curve"])
        resampled[well] = zscore(r)

    wells = list(well_data.keys())
    results = []
    warping_info = {}
    for i in range(len(wells)):
        for j in range(i + 1, len(wells)):
            w1, w2 = wells[i], wells[j]
            a, b = resampled[w1], resampled[w2]
            d, path, radius, paths = banded_dtw_distance(a, b)
            results.append({"pair": f"{w1} vs {w2}", "dtw_distance": d,
                             "sakoe_chiba_radius": radius})
            warping_info[(w1, w2)] = {"a": a, "b": b, "paths": paths,
                                       "path": path, "radius": radius}
            print(f"  {w1} vs {w2}: distance={d:.4f} (band radius={radius} samples)")

    print("\n  Building random-walk null distribution...")
    null_dists = random_walk_null_distances()
    print(f"  Null: mean={null_dists.mean():.4f}, std={null_dists.std():.4f}")

    for r in results:
        z = (r["dtw_distance"] - null_dists.mean()) / (null_dists.std() + 1e-9)
        r["z_vs_null"] = z
        print(f"    {r['pair']}: z-score vs null = {z:+.2f}")

    return pd.DataFrame(results), warping_info, resampled


def plot_dtw_alignment(w1, w2, warping_info, well_data, z_score, out_path):
    """
    Complete DTW visualization: the two curves, the cost matrix with the
    Sakoe-Chiba band edges shaded, the actual warping path traced through
    that corridor, and the two curves connected by their alignment.
    """
    info = warping_info[(w1, w2)]
    a, b, paths, path, radius = info["a"], info["b"], info["paths"], info["path"], info["radius"]
    n = len(a)

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 2.5, 1], width_ratios=[1, 1])

    colors = well_colors(well_data)
    color1, color2 = colors[w1], colors[w2]

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(np.linspace(0, 1, len(a)), a, color=color1, lw=1.2)
    ax_a.set_title(f"{w1} -- INPEFA(GR)", fontsize=10)
    ax_a.set_xlabel("Fractional depth"); ax_a.grid(alpha=0.3)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(np.linspace(0, 1, len(b)), b, color=color2, lw=1.2)
    ax_b.set_title(f"{w2} -- INPEFA(GR)", fontsize=10)
    ax_b.set_xlabel("Fractional depth"); ax_b.grid(alpha=0.3)

    ax_mat = fig.add_subplot(gs[1, :])
    cost = np.array(paths)[1:, 1:]
    cost_display = np.where(np.isinf(cost), np.nan, cost)
    im = ax_mat.imshow(cost_display.T, origin="lower", cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax_mat, label="Cumulative DTW cost (root of accumulated squared cost)")

    upper = [min(n - 1, i + radius) for i in range(n)]
    lower = [max(0, i - radius) for i in range(n)]
    ax_mat.plot(range(n), upper, color="red", lw=1.2, ls="--", label="Sakoe-Chiba band edge")
    ax_mat.plot(range(n), lower, color="red", lw=1.2, ls="--")

    path_i = [p[0] for p in path]
    path_j = [p[1] for p in path]
    ax_mat.plot(path_i, path_j, color="white", lw=2.0, label="Actual warping path")

    ax_mat.set_xlabel(f"{w1} sample index (fractional depth)")
    ax_mat.set_ylabel(f"{w2} sample index (fractional depth)")
    ax_mat.set_title(f"DTW cost matrix -- Sakoe-Chiba radius={radius} samples "
                      f"({SAKOE_CHIBA_FRACTION:.0%}), |i-j| <= {radius} inside the red corridor",
                      fontsize=11)
    ax_mat.legend(loc="upper left", fontsize=9)

    ax_align = fig.add_subplot(gs[2, :])
    offset = 5
    for (i, j) in path[::3]:
        ax_align.plot([i/n, j/len(b)], [a[i], b[j] + offset],
                      color="gray", lw=0.3, alpha=0.5)
    ax_align.plot(np.linspace(0, 1, len(a)), a, color=color1, lw=1.4, label=w1)
    ax_align.plot(np.linspace(0, 1, len(b)), b + offset, color=color2, lw=1.4, label=w2)
    ax_align.set_title(f"Alignment: {w1} vs {w2} (z-score vs random-walk null = {z_score:+.2f})",
                        fontsize=11)
    ax_align.set_xlabel("Fractional depth")
    ax_align.legend(loc="upper right")
    ax_align.grid(alpha=0.3)

    fig.suptitle(f"Complete INPEFA-DTW alignment, GR, Sakoe-Chiba banded -- {w1} vs {w2}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    print(f"  Saved -> {out_path}")


def out_path_for(name):
    """
    Route an output file to csv_out/ or image_output/ by extension,
    creating the folder the first time something is written to it.
    """
    directory = CSV_DIR if str(name).lower().endswith(".csv") else IMAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def write_inpefa_curve(well, df, curve):
    """
    One CSV per well holding just the depth and the oriented long-term INPEFA
    curve -- the same curve the segmentation and the DTW correlation run on --
    so a single well can be picked up in another tool without re-running this.
    """
    out = out_path_for(f"inpefa_curve_{well}.csv")
    pd.DataFrame({"DEPTH": df["DEPTH"].to_numpy(),
                  "INPEFA": np.asarray(curve, dtype=float)}).to_csv(out, index=False)
    print(f"  Saved -> {out}")


def main(data_dir=None):
    all_segments = []
    well_data = {}

    data_dir = Path(data_dir if data_dir is not None else DATA_DIR)
    files = discover_wells(data_dir)
    print(f"Found {len(files)} well(s) in {data_dir}: {', '.join(files)}")

    for well, path in files.items():
        print(f"\n=== {well} (GR, full INPEFA) ===")
        df = load_well(path)
        gr = df["GR"].to_numpy()

        ipfy = inpefa_core(gr)
        curve = orient_to_base_level(ipfy[INPEFA_ORDER], curve="GR")
        write_inpefa_curve(well, df, curve)

        ext_idx = find_extrema(curve)
        print(f"  {len(ext_idx)} extrema detected (smooth_win={SMOOTH_WIN})")

        seg_df = segment_by_extrema(df, curve, ext_idx)
        seg_df = classify_by_angle(seg_df)
        seg_df["WELL"] = well
        all_segments.append(seg_df)
        well_data[well] = {"df": df, "curve": curve, "segments": seg_df}

        counts = seg_df["shape"].value_counts()
        n_symm = seg_df["symmetrical"].sum() // 2
        n_serr = seg_df["serrated"].sum()
        print(f"  {len(seg_df)} segments -> " +
              ", ".join(f"{k}={v}" for k, v in counts.items()))
        print(f"  {n_symm} symmetrical couplets, {n_serr} segments flagged serrated")
        print(f"  Angle range by shape (deg):")
        print(seg_df.groupby("shape")["angle_deg"].agg(["mean", "min", "max", "count"]).round(1))
        print(f"  Thickness by shape (ft):")
        print(seg_df.groupby("shape")["thickness_ft"].agg(["mean", "median", "count"]).round(0))

    all_df = pd.concat(all_segments, ignore_index=True)
    segments_csv = out_path_for("angle_shape_EmeryMyers_segments.csv")
    all_df.to_csv(segments_csv, index=False)
    print(f"\nSaved -> {segments_csv}")

    if len(well_data) < 2:
        print("\nOnly one well -- skipping well-to-well DTW correlation.")
    else:
        dtw_results, warping_info, resampled = run_dtw_correlation(well_data)
        dtw_csv = out_path_for("angle_shape_EmeryMyers_dtw.csv")
        dtw_results.to_csv(dtw_csv, index=False)
        print(f"\nSaved -> {dtw_csv}")

        print("\nGenerating complete DTW alignment plots...")
        for _, row in dtw_results.iterrows():
            w1, w2 = row["pair"].split(" vs ")
            out_path = out_path_for(f"dtw_alignment_EmeryMyers_{w1}_vs_{w2}.png")
            plot_dtw_alignment(w1, w2, warping_info, well_data, row["z_vs_null"], out_path)

        plot_dtw_matrix(dtw_results, well_data)

    plot_wells(well_data)
    plot_angle_distribution(all_df)


def dtw_matrices(dtw_results, wells):
    """
    Fold the pairwise DTW results into two square well-by-well matrices,
    laid out like a confusion matrix: distance (lower = more alike) and
    z-score against the random-walk null (more negative = more alike than
    two unrelated integrated curves would be).

    Both are symmetric -- DTW with a symmetric band and a symmetric step
    pattern gives d(a, b) = d(b, a). The distance diagonal is 0 by
    definition (a curve against itself); the z diagonal is left as NaN,
    since "how surprising is a well matching itself" is not a question the
    null can answer.
    """
    idx = {w: i for i, w in enumerate(wells)}
    n = len(wells)
    dist = np.full((n, n), np.nan)
    zmat = np.full((n, n), np.nan)
    np.fill_diagonal(dist, 0.0)
    for _, r in dtw_results.iterrows():
        w1, w2 = r["pair"].split(" vs ")
        i, j = idx[w1], idx[w2]
        dist[i, j] = dist[j, i] = r["dtw_distance"]
        zmat[i, j] = zmat[j, i] = r["z_vs_null"]
    return dist, zmat


def _annotate_matrix(ax, mat, wells, fmt, na_text):
    """Write each cell's value on top of the heatmap, like a confusion matrix."""
    finite = mat[np.isfinite(mat)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0.0
    # shrink the text as the grid grows so cells never collide
    fs = 10 if len(wells) <= 4 else (8 if len(wells) <= 7 else 6.5)
    for i in range(len(wells)):
        for j in range(len(wells)):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, na_text, ha="center", va="center",
                        color="0.45", fontsize=fs)
            else:
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if v < mid else "black", fontsize=fs)
    ax.set_xticks(range(len(wells))); ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.set_yticks(range(len(wells))); ax.set_yticklabels(wells)
    ax.set_xticks(np.arange(len(wells) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(wells) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)


def plot_dtw_matrix(dtw_results, well_data,
                    out_path=None):
    """
    Well-by-well correlation-strength matrix: the pairwise DTW results
    arranged as a square grid so every well can be read against every other
    at a glance, the way a confusion matrix is read.
    """
    out_path = out_path or out_path_for("dtw_correlation_matrix_EmeryMyers.png")
    wells = list(well_data.keys())
    dist, zmat = dtw_matrices(dtw_results, wells)

    # both dimensions grow with the well count so the cells stay readable
    n = len(wells)
    fig, axes = plt.subplots(1, 2, figsize=(6.0 + 1.15 * n, 3.2 + 0.55 * n))

    # The diagonal is 0 by construction and carries no information; leaving it
    # in the color scale would squash the range that actually matters, so the
    # colors span the off-diagonal (well-to-well) distances only.
    off_diag = dist.copy()
    np.fill_diagonal(off_diag, np.nan)
    im0 = axes[0].imshow(np.ma.masked_invalid(off_diag), cmap="viridis_r")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, label="Banded DTW distance (RMS per aligned pair)")
    axes[0].set_title("DTW distance\n(lower = more similar)", fontsize=11)
    _annotate_matrix(axes[0], off_diag, wells, ".4f", "0")

    zlim = np.nanmax(np.abs(zmat)) if np.isfinite(zmat).any() else 1.0
    im1 = axes[1].imshow(np.ma.masked_invalid(zmat), cmap="RdYlGn_r",
                         vmin=-zlim, vmax=zlim)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, label="z-score vs random-walk null")
    axes[1].set_title("Correlation strength vs null\n(more negative = stronger)", fontsize=11)
    _annotate_matrix(axes[1], zmat, wells, "+.2f", "n/a")

    fig.suptitle(f"Well-to-well INPEFA(GR) DTW correlation matrix -- "
                 f"Sakoe-Chiba band {SAKOE_CHIBA_FRACTION:.0%}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    print(f"Saved -> {out_path}")


def plot_wells(well_data, out_path=None):
    out_path = out_path or out_path_for("angle_shape_EmeryMyers_facies.png")
    wells = list(well_data.keys())
    shape_cmap = ListedColormap([SHAPE_COLORS["BELL"], SHAPE_COLORS["FUNNEL"],
                                  SHAPE_COLORS["CYLINDRICAL"]])
    shape_to_int = {"BELL": 0, "FUNNEL": 1, "CYLINDRICAL": 2}

    # width scales with the well count (5.4 in per well keeps the 3-well
    # figure the size it has always been); the floor keeps the suptitle from
    # being clipped when only one or two wells are in the folder
    fig_width = max(11.0, 5.4 * len(wells))
    fig, axes = plt.subplots(1, len(wells) * 3, figsize=(fig_width, 13))
    for i, well in enumerate(wells):
        data = well_data[well]
        df = data["df"]
        seg_df = data["segments"]
        depth = df["DEPTH"].to_numpy()

        ax_gr, ax_curve, ax_shape = axes[i*3], axes[i*3+1], axes[i*3+2]

        ax_gr.plot(df["GR"], depth, color="green", lw=0.3)
        ax_gr.set_xlim(0, 200); ax_gr.invert_yaxis()
        ax_gr.set_title(f"{well}\nGR (raw)", fontsize=9); ax_gr.grid(alpha=0.3)
        if i == 0:
            ax_gr.set_ylabel("DEPTH (ft)")

        ax_curve.plot(data["curve"], depth, color="black", lw=0.5)
        for _, row in seg_df.iterrows():
            ax_curve.axhline(row["top_depth"], color="gray", lw=0.2, alpha=0.4)
        ax_curve.invert_yaxis()
        ax_curve.set_title("Full INPEFA\n(GR, order 1)", fontsize=9); ax_curve.grid(alpha=0.3)
        ax_curve.set_yticklabels([])

        shape_arr = np.zeros(len(depth), dtype=int)
        for _, row in seg_df.iterrows():
            mask = (depth >= row["top_depth"]) & (depth <= row["base_depth"])
            shape_arr[mask] = shape_to_int[row["shape"]]
        ax_shape.imshow(shape_arr.reshape(-1, 1), aspect="auto",
                        extent=[0, 1, depth.max(), depth.min()],
                        cmap=shape_cmap, vmin=0, vmax=2)
        for _, row in seg_df[seg_df["symmetrical"]].iterrows():
            ax_shape.axhline(row["top_depth"], color="yellow", lw=1.0, alpha=0.9)
        ax_shape.set_title("Shape\n(GR, full INPEFA)", fontsize=8)
        ax_shape.set_xticks([]); ax_shape.set_yticklabels([])

    handles = [plt.Rectangle((0,0),1,1, color=c) for c in SHAPE_COLORS.values()]
    fig.legend(handles, SHAPE_COLORS.keys(), loc="lower center", ncol=3, fontsize=10)
    fig.suptitle("Angle-based shape typing -- FULL INPEFA on GR only "
                 "(Emery & Myers 1996: BELL/FUNNEL/CYLINDRICAL)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    print(f"\nSaved -> {out_path}")


def plot_angle_distribution(all_df, out_path=None):
    out_path = out_path or out_path_for("angle_distribution_EmeryMyers.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for shape, color in SHAPE_COLORS.items():
        vals = all_df.loc[all_df["shape"] == shape, "angle_deg"]
        ax.hist(vals, bins=25, alpha=0.6, label=shape, color=color)
    ax.axvline(-BELL_FUNNEL_LOWER_DEG, color="black", lw=0.8, ls="--")
    ax.axvline(BELL_FUNNEL_LOWER_DEG, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Segment angle (degrees, normalized 100ft vs z-scored full INPEFA)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of extrema-segment angles (GR, full INPEFA), "
                 + " + ".join(all_df["WELL"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    # optional: python mainpefa.py /path/to/other/las/folder
    main(sys.argv[1] if len(sys.argv) > 1 else None)
