"""
Spectral flow cytometry panel selection tool.

Given per-fluorophore spectral signatures (measured on a spectral cytometer,
e.g. CytoFLEX Mosaic), select the N fluorophores (at most M of which may be
HaloTag/SNAPtag-type dyes) that are maximally "spectrally distinct" from
each other.

"Distinct" combines two goals:
  - large AVERAGE (or median) pairwise distance between all chosen colors
  - large MINIMUM pairwise distance (no two colors that are hard to unmix)

This is a variant of the "max-min diversity" / p-dispersion problem, which
is NP-hard in general, so for anything but tiny panels we use a greedy
farthest-point construction followed by local-search swapping, repeated
with many random restarts. Small cases are solved exactly by brute force.
"""

import itertools
import math
import random
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# 1. Loading
# --------------------------------------------------------------------------

def load_spectra(path_or_buffer, sheet_name=0, control_name="HEK293T"):
    """
    Load a CytExpert/CytoFLEX-style "unmixing matrix" export.

    `path_or_buffer` can be a file path (str/Path) OR a file-like object
    (e.g. what st.file_uploader() returns) -- pandas handles both.

    Returns
    -------
    spectra : DataFrame (channels x fluorophores), control column removed
    control : Series, the unstained-control spectrum (kept for reference)
    """
    raw = pd.read_excel(path_or_buffer, sheet_name=sheet_name, header=None)

    # Row 3 (0-indexed) holds the column headers ("Channel", "mTagBFP2%", ...)
    header_row = raw.iloc[3, :].tolist()
    data = raw.iloc[4:, :].reset_index(drop=True)

    channels = data.iloc[:, 0].astype(str).tolist()
    names = [str(h).rstrip("%").strip() for h in header_row[1:]]

    values = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    values.columns = names
    values.index = channels

    control = None
    if control_name in values.columns:
        control = values[control_name].copy()
        values = values.drop(columns=[control_name])

    return values, control


def classify_fluorophores(spectra, dye_prefixes=("JF",)):
    """
    Split fluorophore names into "FP" (fluorescent protein) vs "dye"
    (HaloTag/SNAPtag-type, e.g. Janelia Fluor "JF..."/"JFX..." dyes) groups,
    based on name prefix. Adjust `dye_prefixes` if your dye names differ.
    """
    dye_names = [c for c in spectra.columns if c.upper().startswith(dye_prefixes)]
    fp_names = [c for c in spectra.columns if c not in dye_names]
    return fp_names, dye_names


def sort_names_by_peak_channel(spectra, names):
    """
    Sort fluorophore names by which channel each one peaks in, in the
    cytometer's natural channel order (as given by spectra.index -- e.g.
    all V channels, then all B, then Y, then R, since that's the order
    the channels appear in the exported unmixing matrix).
    """
    channel_order = {ch: i for i, ch in enumerate(spectra.index)}
    peak_channel = {n: spectra[n].idxmax() for n in names}
    return sorted(names, key=lambda n: channel_order[peak_channel[n]])


# --------------------------------------------------------------------------
# 1b. Channel wavelength reference
#
# Beckman publishes each laser's detector count and overall wavelength
# range (e.g. "Violet 405 nm: 20 detectors, 420-950 nm") but not the exact
# per-detector cut points, so DEFAULT_LASER_CONFIG's ranges are divided
# evenly across each laser's detector count as an APPROXIMATION. Good
# enough to orient a viewer (e.g. "R4 is roughly in the 700s nm"), not
# a substitute for the instrument's real filter specs if exact bandpass
# edges matter for your analysis.
# --------------------------------------------------------------------------

DEFAULT_LASER_CONFIG = {
    "V": {"label": "Violet",       "laser_nm": 405, "n_detectors": 20, "range_nm": (420, 950)},
    "B": {"label": "Blue",         "laser_nm": 488, "n_detectors": 16, "range_nm": (498, 950)},
    "Y": {"label": "Yellow-Green", "laser_nm": 561, "n_detectors": 12, "range_nm": (567, 950)},
    "R": {"label": "Red",          "laser_nm": 638, "n_detectors": 10, "range_nm": (649, 950)},
}


def channel_wavelength_table(spectra, laser_config=None):
    """
    Build a per-channel wavelength-range reference table for the channels
    in `spectra.index` (e.g. "V1", "V2", ..., "B1", ...), based on
    `laser_config` (defaults to DEFAULT_LASER_CONFIG, the CytoFLEX mosaic
    V-B-Y-R spec). Ranges are evenly divided across each laser's published
    range -- see module note above re: approximation.

    Returns a DataFrame indexed by channel name with columns:
    laser, laser_nm, detector_index, low_nm, high_nm, center_nm
    """
    laser_config = laser_config or DEFAULT_LASER_CONFIG
    rows = []
    for ch in spectra.index:
        m = re.match(r"([A-Za-z]+)(\d+)", str(ch))
        if not m:
            rows.append({"channel": ch, "laser": None, "laser_nm": None,
                         "detector_index": None, "low_nm": None, "high_nm": None,
                         "center_nm": None})
            continue
        prefix, num = m.group(1).upper(), int(m.group(2))
        cfg = laser_config.get(prefix)
        if cfg is None:
            rows.append({"channel": ch, "laser": prefix, "laser_nm": None,
                         "detector_index": num, "low_nm": None, "high_nm": None,
                         "center_nm": None})
            continue
        lo, hi = cfg["range_nm"]
        edges = np.linspace(lo, hi, cfg["n_detectors"] + 1)
        low, high = edges[num - 1], edges[num]
        rows.append({
            "channel": ch,
            "laser": cfg["label"],
            "laser_nm": cfg["laser_nm"],
            "detector_index": num,
            "low_nm": round(float(low), 1),
            "high_nm": round(float(high), 1),
            "center_nm": round(float((low + high) / 2), 1),
        })
    return pd.DataFrame(rows).set_index("channel")


# --------------------------------------------------------------------------
# 2. Distance metric
# --------------------------------------------------------------------------

def cosine_distance_matrix(spectra, names):
    """
    Pairwise cosine distance (1 - cosine similarity) between spectral shapes,
    after L2-normalizing each spectrum. This measures how distinguishable
    the *shape* of two spectra are, independent of overall brightness --
    the standard notion of spectral similarity/overlap used by spectral
    cytometry software (higher distance = easier to unmix).

    Returns an (n x n) numpy array in the order of `names`.
    """
    mat = spectra[names].to_numpy(dtype=float)
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    unit = mat / norms
    sim = unit.T @ unit
    sim = np.clip(sim, -1.0, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


# --------------------------------------------------------------------------
# 3. Scoring
# --------------------------------------------------------------------------

def _pairwise_vals(dist, idx):
    idx = list(idx)
    sub = dist[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return sub[iu]


def subset_score(dist, idx, w_avg=0.5, w_min=0.5, use_median=False):
    """
    Combined objective for a candidate panel (given as a list of indices
    into the distance matrix). Higher is better.

    score = w_avg * (mean or median pairwise distance) + w_min * (min pairwise distance)

    w_avg=1, w_min=0  -> pure "maximize average separation"
    w_avg=0, w_min=1  -> pure "maximize worst-case separation" (classic p-dispersion)
    """
    vals = _pairwise_vals(dist, idx)
    center = np.median(vals) if use_median else np.mean(vals)
    return w_avg * center + w_min * vals.min(), center, vals.min()


# --------------------------------------------------------------------------
# 4. Exact (brute force) search -- only for small combinatorial spaces
#
# All functions below search only over a "free" candidate pool
# (fluorophores not hard-coded and not excluded) and always prepend any
# `fixed_idx` (required/hard-coded fluorophores) to every candidate set.
# `remaining_N`/`remaining_M` are what's left to choose after accounting
# for the fixed picks (remaining_N = N - len(fixed), remaining_M =
# M - number of fixed picks that are dyes).
# --------------------------------------------------------------------------

def _n_combinations(n_fp, n_dye, remaining_N, remaining_M):
    total = 0
    for k in range(0, min(remaining_M, remaining_N) + 1):
        n_fp_needed = remaining_N - k
        if 0 <= n_fp_needed <= n_fp and k <= n_dye:
            total += math.comb(n_dye, k) * math.comb(n_fp, n_fp_needed)
    return total


def _brute_force(dist, free_fp_idx, free_dye_idx, remaining_N, remaining_M,
                  w_avg, w_min, use_median, fixed_idx=()):
    fixed_idx = tuple(fixed_idx)
    best_idx, best = None, (-np.inf, None, None)
    for k in range(0, min(remaining_M, remaining_N) + 1):
        n_fp_needed = remaining_N - k
        if n_fp_needed < 0 or n_fp_needed > len(free_fp_idx) or k > len(free_dye_idx):
            continue
        for dye_combo in itertools.combinations(free_dye_idx, k):
            for fp_combo in itertools.combinations(free_fp_idx, n_fp_needed):
                idx = fixed_idx + dye_combo + fp_combo
                sc, avg, mn = subset_score(dist, idx, w_avg, w_min, use_median)
                if sc > best[0]:
                    best_idx, best = idx, (sc, avg, mn)
    return list(best_idx), best


# --------------------------------------------------------------------------
# 5. Heuristic search -- greedy farthest-point init + local swap search,
#    with many random restarts. `fixed_idx` (required fluorophores) are
#    seeded into every initial solution and are never removed by swaps;
#    `free_*_idx` are the only pool considered for the remaining slots.
# --------------------------------------------------------------------------

def _greedy_init(dist, free_idx, target_N, dye_set, M, rng, fixed_idx=()):
    """Farthest-point-style greedy construction respecting the dye cap.

    `dye_set` should contain ALL dye indices (fixed + free) so the cap
    is enforced correctly against the whole panel, not just the free part.
    """
    sel = list(fixed_idx)
    candidates = list(free_idx)
    if not sel:
        sel = [rng.choice(candidates)]
    while len(sel) < target_N:
        n_dye = sum(1 for i in sel if i in dye_set)
        avail = [c for c in candidates
                 if c not in sel and not (c in dye_set and n_dye >= M)]
        if not avail:
            break
        # pick the candidate maximizing its minimum distance to the current set
        best_c, best_val = None, -1.0
        for c in avail:
            d = min(dist[c, s] for s in sel)
            if d > best_val:
                best_val, best_c = d, c
        sel.append(best_c)
    return sel


def _random_init(free_fp_idx, free_dye_idx, target_N, M, rng, fixed_idx=(), n_fixed_dye=0):
    remaining_N = target_N - len(fixed_idx)
    remaining_M = max(0, min(M - n_fixed_dye, remaining_N))
    k = rng.randint(0, remaining_M)
    k = min(k, len(free_dye_idx))
    n_fp_needed = remaining_N - k
    if n_fp_needed > len(free_fp_idx):
        n_fp_needed = min(remaining_N, len(free_fp_idx))
        k = remaining_N - n_fp_needed
    sel = list(fixed_idx) + rng.sample(list(free_dye_idx), k) + rng.sample(list(free_fp_idx), n_fp_needed)
    return sel


def _local_search(dist, sel, free_idx, dye_set, M, w_avg, w_min, use_median,
                   fixed_set=frozenset()):
    sel = set(sel)
    universe = set(free_idx)  # candidates available to swap IN (fixed excluded by construction)
    cur_score, cur_avg, cur_min = subset_score(dist, sel, w_avg, w_min, use_median)
    improved = True
    while improved:
        improved = False
        outside = universe - sel
        for out_c in list(sel):
            if out_c in fixed_set:
                continue  # never swap out a required/hard-coded fluorophore
            n_dye_without = sum(1 for i in sel if i in dye_set) - (1 if out_c in dye_set else 0)
            for in_c in outside:
                if in_c in dye_set and n_dye_without >= M:
                    continue
                new_sel = (sel - {out_c}) | {in_c}
                sc, avg, mn = subset_score(dist, new_sel, w_avg, w_min, use_median)
                if sc > cur_score + 1e-9:
                    sel, cur_score, cur_avg, cur_min = new_sel, sc, avg, mn
                    improved = True
                    break
            if improved:
                break
    return sel, cur_score, cur_avg, cur_min


def _heuristic_search(dist, free_fp_idx, free_dye_idx, N, M, w_avg, w_min, use_median,
                       n_restarts, seed, fixed_idx=(), all_dye_idx=None):
    rng = random.Random(seed)
    free_idx = free_fp_idx + free_dye_idx
    dye_set = set(all_dye_idx if all_dye_idx is not None else free_dye_idx)
    fixed_set = set(fixed_idx)
    n_fixed_dye = sum(1 for i in fixed_idx if i in dye_set)
    best = None
    for r in range(n_restarts):
        if r % 2 == 0:
            init = _greedy_init(dist, free_idx, N, dye_set, M, rng, fixed_idx=fixed_idx)
        else:
            init = _random_init(free_fp_idx, free_dye_idx, N, M, rng,
                                 fixed_idx=fixed_idx, n_fixed_dye=n_fixed_dye)
        if len(init) < N:
            continue
        sel, sc, avg, mn = _local_search(dist, init, free_idx, dye_set, M,
                                          w_avg, w_min, use_median, fixed_set=fixed_set)
        if best is None or sc > best[1][0]:
            best = (list(sel), (sc, avg, mn))
    return best


# --------------------------------------------------------------------------
# 6. Public API
# --------------------------------------------------------------------------

def select_panel(spectra, N, M, fp_names=None, dye_names=None,
                  required_names=None, excluded_names=None,
                  w_avg=0.5, w_min=0.5, use_median=False,
                  n_restarts=200, brute_force_limit=200_000, seed=0,
                  sort_output_by="peak_channel"):
    """
    Select the best N-color panel (at most M dyes) from `spectra`.

    Parameters
    ----------
    spectra : DataFrame (channels x fluorophores), e.g. from load_spectra()
    N : total number of colors in the panel
    M : maximum number of them that may be HaloTag/SNAPtag-type dyes
    fp_names, dye_names : optional explicit lists; auto-detected via
        classify_fluorophores() if not given
    required_names : fluorophores to hard-code into the panel (e.g. dyes/FPs
        already committed to in an existing system). These always count
        toward both N and, if they are dyes, toward M. The search only
        optimizes the remaining N - len(required_names) slots.
    excluded_names : fluorophores to remove from consideration entirely
        (e.g. ones you know you can't use). Cannot overlap with
        required_names.
    w_avg, w_min : weights on (mean/median pairwise distance) vs
        (minimum pairwise distance) in the objective. Must be >= 0.
    use_median : use median instead of mean for the "average" term
        (more robust to one outlier pair)
    n_restarts : random restarts for the heuristic search
    brute_force_limit : if the number of valid subsets left to search is at
        or below this, solve exactly instead of heuristically
    seed : RNG seed for reproducibility
    sort_output_by : "peak_channel" (default) to order the returned panel
        by each fluorophore's peak channel in cytometer order (V's, B's,
        Y's, R's); "required_first" to instead list required_names first
        then the rest as found.

    Returns
    -------
    dict with keys: 'panel' (list of names), 'score', 'mean_dist' (or
        median), 'min_dist', 'dist_matrix' (DataFrame, panel x panel),
        'exact' (bool)
    """
    if fp_names is None or dye_names is None:
        auto_fp, auto_dye = classify_fluorophores(spectra)
        fp_names = fp_names if fp_names is not None else list(auto_fp)
        dye_names = dye_names if dye_names is not None else list(auto_dye)
    else:
        fp_names, dye_names = list(fp_names), list(dye_names)

    required_names = list(required_names) if required_names else []
    excluded_names = list(excluded_names) if excluded_names else []

    overlap = set(required_names) & set(excluded_names)
    if overlap:
        raise ValueError(f"Names cannot be both required and excluded: {sorted(overlap)}")

    # Remove excluded fluorophores from the candidate pools entirely.
    fp_names = [n for n in fp_names if n not in excluded_names]
    dye_names = [n for n in dye_names if n not in excluded_names]

    all_names = fp_names + dye_names
    unknown_required = [n for n in required_names if n not in all_names]
    if unknown_required:
        raise ValueError(f"required_names not found among available fluorophores "
                          f"(check spelling / not accidentally excluded): {unknown_required}")

    if N > len(all_names):
        raise ValueError(f"N={N} exceeds total available fluorophores "
                          f"({len(fp_names)} FP + {len(dye_names)} dye, after exclusions)")
    if M > len(dye_names):
        raise ValueError(f"M={M} exceeds available dyes ({len(dye_names)}, after exclusions)")
    if len(required_names) > N:
        raise ValueError(f"{len(required_names)} required_names exceeds panel size N={N}")

    dye_name_set = set(dye_names)
    n_required_dye = sum(1 for n in required_names if n in dye_name_set)
    if n_required_dye > M:
        raise ValueError(f"required_names includes {n_required_dye} dye(s) but M={M}")

    remaining_N = N - len(required_names)
    remaining_M = M - n_required_dye
    if N - M > len(fp_names):
        raise ValueError(f"Need at least {N - M} FPs but only "
                          f"{len(fp_names)} are available (after exclusions)")

    dist = cosine_distance_matrix(spectra, all_names)
    fp_idx_all = list(range(len(fp_names)))
    dye_idx_all = list(range(len(fp_names), len(fp_names) + len(dye_names)))
    name_to_idx = {n: i for i, n in enumerate(all_names)}

    fixed_idx = [name_to_idx[n] for n in required_names]
    fixed_set = set(fixed_idx)
    free_fp_idx = [i for i in fp_idx_all if i not in fixed_set]
    free_dye_idx = [i for i in dye_idx_all if i not in fixed_set]

    n_combos = _n_combinations(len(free_fp_idx), len(free_dye_idx), remaining_N, remaining_M)
    exact = n_combos <= brute_force_limit

    if remaining_N == 0:
        # Nothing left to search -- the panel is exactly the required set.
        idx = fixed_idx
        score, center, mn = subset_score(dist, idx, w_avg, w_min, use_median)
        exact = True
    elif exact:
        idx, (score, center, mn) = _brute_force(
            dist, free_fp_idx, free_dye_idx, remaining_N, remaining_M,
            w_avg, w_min, use_median, fixed_idx=fixed_idx)
    else:
        idx, (score, center, mn) = _heuristic_search(
            dist, free_fp_idx, free_dye_idx, N, M, w_avg, w_min, use_median,
            n_restarts, seed, fixed_idx=fixed_idx, all_dye_idx=dye_idx_all)

    # Order the panel for display. Default: by peak channel (V's, then B's,
    # then Y's, then R's) so the printed panel/heatmap read left-to-right
    # the way the spectrum would scan. Pass sort_output_by="required_first"
    # to instead list required_names first (previous behavior).
    if sort_output_by == "peak_channel":
        idx = sort_names_by_peak_channel(spectra, [all_names[i] for i in idx])
        idx = [name_to_idx[n] for n in idx]
    else:
        idx = list(fixed_idx) + [i for i in idx if i not in fixed_set]
    panel = [all_names[i] for i in idx]
    sub_dist = pd.DataFrame(dist[np.ix_(idx, idx)], index=panel, columns=panel)

    return {
        "panel": panel,
        "score": score,
        "mean_dist" if not use_median else "median_dist": center,
        "min_dist": mn,
        "dist_matrix": sub_dist,
        "exact": exact,
        "n_combinations_considered": n_combos,
        "required_names": required_names,
        "excluded_names": excluded_names,
    }


def summarize_panel(result):
    lines = []
    lines.append(f"Panel ({len(result['panel'])} colors): {', '.join(result['panel'])}")
    if result.get("required_names"):
        lines.append(f"  (required: {', '.join(result['required_names'])})")
    if result.get("excluded_names"):
        lines.append(f"  (excluded from search: {', '.join(result['excluded_names'])})")
    key = "mean_dist" if "mean_dist" in result else "median_dist"
    lines.append(f"  {key}: {result[key]:.4f}   min_dist: {result['min_dist']:.4f}   "
                 f"score: {result['score']:.4f}")
    lines.append(f"  solved {'exactly' if result['exact'] else 'heuristically'} "
                 f"({result['n_combinations_considered']:,} combinations "
                 f"{'evaluated' if result['exact'] else 'in search space'})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6b. Extras: visualization + alternative panels
# --------------------------------------------------------------------------

def _turbo_hex_colors(n):
    """Sample n evenly-spaced colors from matplotlib's 'turbo' colormap as hex strings."""
    cmap = plt.get_cmap("turbo", n)
    return [
        "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in cmap(i)[:3])
        for i in range(n)
    ]


def plot_spectra_interactive(spectra, panel, channel_table=None, sort_by_peak=True):
    """
    UNDER CONSTRUCTION: interactive (Plotly) version of plot_spectra().
    Hovering shows a shared gray vertical guide line across all traces
    (via Plotly's "spike lines") plus the wavelength range of the
    channel under the cursor, alongside each fluorophore's value there.

    Returns a plotly.graph_objects.Figure -- render with st.plotly_chart(fig).
    """
    import plotly.graph_objects as go

    names = sort_names_by_peak_channel(spectra, panel) if sort_by_peak else list(panel)
    channels = spectra.index.tolist()
    x = list(range(len(channels)))

    if channel_table is None:
        channel_table = channel_wavelength_table(spectra)
    low = channel_table.loc[channels, "low_nm"].to_numpy()
    high = channel_table.loc[channels, "high_nm"].to_numpy()
    range_text = [
        f"{lo:.0f}\u2013{hi:.0f} nm" if not np.isnan(lo) else "n/a"
        for lo, hi in zip(low, high)
    ]

    colors = _turbo_hex_colors(len(names))
    fig = go.Figure()

    # Invisible reference trace (sits below the visible y-range) purely so
    # the unified hover box also shows the wavelength range of the channel
    # under the cursor, in addition to each fluorophore's per-trace value.
    fig.add_trace(go.Scatter(
        x=x, y=[-5] * len(x), mode="lines", name="Wavelength range",
        line=dict(width=0),
        customdata=range_text,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))

    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=x, y=spectra[name].to_numpy(), mode="lines", name=name,
            line=dict(width=3, color=colors[i]),
            hovertemplate="%{y:.1f}<extra></extra>",
        ))

    # Laser-group boundaries (V/B/Y/R), detected from channel-name prefix.
    prefixes = [ch[0] for ch in channels]
    for i in range(1, len(prefixes)):
        if prefixes[i] != prefixes[i - 1]:
            fig.add_vline(x=i - 0.5, line=dict(color="black", width=3, dash="solid"))

    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(
            title="Channel",
            tickmode="array", tickvals=x, ticktext=channels, tickangle=90,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor="rgba(120,120,120,0.5)", spikethickness=2, spikedash="solid",
            range=[-0.5, len(channels) - 0.5],
        ),
        yaxis=dict(title="Normalized signal (% of peak)", range=[0, 105]),
        title=f"Spectral signatures (interactive) -- {len(names)}-color panel",
        height=550,
        margin=dict(r=160),
    )
    return fig


def plot_spectra(spectra, panel, ax=None, save_path=None, sort_by_peak=True):
    """
    Overlapping line chart of the raw spectral signatures for the
    fluorophores in `panel` -- channel (in cytometer order) on the x-axis,
    normalized signal (0-100) on the y-axis, one line per fluorophore.

    spectra : DataFrame (channels x fluorophores), e.g. from load_spectra()
    panel : list of fluorophore names to plot (e.g. result['panel'])
    sort_by_peak : if True, order the legend/line-plotting by peak channel
        (matches sort_names_by_peak_channel) so the legend reads roughly
        left-to-right in the same order the peaks appear on the plot.
    """
    names = sort_names_by_peak_channel(spectra, panel) if sort_by_peak else list(panel)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    channels = spectra.index.tolist()
    x = np.arange(len(channels))
    colors = plt.get_cmap("turbo", len(names))

    for i, name in enumerate(names):
        ax.plot(x, spectra[name].to_numpy(), label=name, color=colors(i), linewidth=2)

    # Light vertical guides between laser groups (V/B/Y/R), detected from
    # the channel name prefix rather than hard-coded, so this still works
    # if your instrument config differs.
    prefixes = [ch[0] for ch in channels]
    boundaries = [i for i in range(1, len(prefixes)) if prefixes[i] != prefixes[i - 1]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="black", linewidth=2, linestyle="-")

    # Show every other tick label to avoid crowding
    ax.set_xticks(x)
    ax.set_xticklabels([ch if i % 2 == 0 else "" for i, ch in enumerate(channels)],
                        rotation=90, fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Normalized signal (% of peak)")
    ax.set_title(f"Spectral signatures -- {len(names)}-color panel")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)
    ax.set_xlim(-0.5, len(channels) - 0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def plot_panel(result, fig=None, ax=None, save_path=None):
    """Heatmap of pairwise distances for a selected panel."""
    sub = result["dist_matrix"]
    if (fig is None) and (ax is None):
        fig, ax = plt.subplots(figsize=(0.6 * len(sub) + 2.5, 0.6 * len(sub) + 2))
    im = ax.imshow(sub.values, cmap="viridis_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(sub)))
    ax.set_yticks(range(len(sub)))
    ax.set_xticklabels(sub.columns, rotation=90)
    ax.set_yticklabels(sub.index)
    for i in range(len(sub)):
        for j in range(len(sub)):
            val = sub.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val >= 0.45 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="pairwise distance (1 - cosine similarity)")
    ax.set_title(f"Panel pairwise distances\nscore={result['score']:.3f}, "
                 f"min dist={result['min_dist']:.3f}")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def top_k_panels(spectra, N, M, k=5, fp_names=None, dye_names=None,
                  required_names=None, excluded_names=None,
                  w_avg=0.5, w_min=0.5, use_median=False,
                  n_restarts=200, n_seeds=8, seed0=0):
    """
    Run the search from several random seeds and return up to k distinct
    panels, best first -- useful to see how much better the top pick is
    than reasonable alternatives (i.e. how "peaked" the optimum is).
    """
    seen = {}
    for s in range(seed0, seed0 + n_seeds):
        r = select_panel(spectra, N, M, fp_names=fp_names, dye_names=dye_names,
                          required_names=required_names, excluded_names=excluded_names,
                          w_avg=w_avg, w_min=w_min, use_median=use_median,
                          n_restarts=n_restarts, seed=s)
        key = tuple(sorted(r["panel"]))
        if key not in seen or r["score"] > seen[key]["score"]:
            seen[key] = r
    ranked = sorted(seen.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:k]
