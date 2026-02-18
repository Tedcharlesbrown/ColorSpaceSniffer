
import numpy as np
import PyOpenColorIO as OCIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import os
from processing import extract_frames, load_image, STILL_IMAGE_EXTENSIONS


# Same IDT list used in main.py — keep in sync
IDTs = [
    "D-Log D-Gamut",
    "Apple Log",
    "ARRI LogC3 (EI800)",
    "ARRI LogC4",
    "BMDFilm WideGamut Gen5",
    "DaVinci Intermediate WideGamut",
    "CanonLog2 CinemaGamut D55",
    "CanonLog3 CinemaGamut D55",
    "V-Log V-Gamut",
    "Log3G10 REDWideGamutRGB",
    "S-Log3 S-Gamut3",
    "S-Log3 S-Gamut3.Cine",
    "S-Log3 Venice S-Gamut3",
    "S-Log3 Venice S-Gamut3.Cine",
]

BINS = 256
LUMA_WEIGHTS = np.float32([0.2126, 0.7152, 0.0722])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scene_linear_name(config):
    try:
        return config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        return "ACES2065-1"


def _make_processor(config, src, dst):
    return config.getProcessor(src, dst).getDefaultCPUProcessor()


# ---------------------------------------------------------------------------
# Encoding-curve generation (via OCIO)
# ---------------------------------------------------------------------------

def generate_encoding_curve(config, idt_name, steps=4096):
    """Encode a linear ramp through the inverse IDT (scene-linear -> encoded).

    Returns a 1-D float32 array: index = linear intensity, value = encoded code value.
    """
    cpu = _make_processor(config, _scene_linear_name(config), idt_name)
    ramp = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    rgb = np.stack([ramp, ramp, ramp], axis=-1).reshape(1, steps, 3).copy()
    cpu.applyRGB(rgb)
    return np.dot(rgb[0], LUMA_WEIGHTS)


def generate_reference_anchors(config, idt_name):
    """Encode known scene-linear values to get reference anchor points.

    Returns [black, mid-grey, shoulder] as encoded code values.
    0.0  -> encoding floor (black level)
    0.18 -> mid-grey
    0.9  -> shoulder / near-highlight
    """
    cpu = _make_processor(config, _scene_linear_name(config), idt_name)
    vals = np.float32([0.0, 0.18, 0.9])
    rgb = np.stack([vals, vals, vals], axis=-1).reshape(1, 3, 3).copy()
    cpu.applyRGB(rgb)
    return np.dot(rgb[0], LUMA_WEIGHTS)


# ---------------------------------------------------------------------------
# Footage analysis
# ---------------------------------------------------------------------------

def footage_cdf_averaged(frames, bins=BINS):
    """Per-frame CDF averaged across frames — smooths scene-dependent variation."""
    accum = np.zeros(bins, dtype=np.float64)
    for frame in frames:
        luma = np.dot(frame.astype(np.float32), LUMA_WEIGHTS)
        hist, _ = np.histogram(luma.ravel(), bins=bins, range=(0.0, 1.0))
        total = hist.sum()
        if total > 0:
            accum += np.cumsum(hist.astype(np.float64)) / total
    return accum / len(frames)


def footage_histogram(frames, bins=BINS):
    """Normalized luminance histogram pooled across all frames."""
    all_luma = []
    for frame in frames:
        luma = np.dot(frame.astype(np.float32), LUMA_WEIGHTS)
        all_luma.append(luma.ravel())
    all_luma = np.concatenate(all_luma)
    hist, _ = np.histogram(all_luma, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def detect_footage_anchors(cdf, bin_edges):
    """Detect black level, mid-grey, and shoulder from the footage CDF.

    Black  = code value at which CDF first reaches 2%  (encoding floor)
    Mid    = code value at which CDF reaches 50%        (median)
    Shoulder = code value at which CDF reaches 95%      (near-highlight)
    """
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    n = len(centers)
    thresholds = [0.02, 0.50, 0.95]
    anchors = np.zeros(3, dtype=np.float64)
    for j, t in enumerate(thresholds):
        idx = int(np.searchsorted(cdf, t))
        anchors[j] = centers[min(idx, n - 1)]
    return anchors


# ---------------------------------------------------------------------------
# CDF / histogram from the encoding curve
# ---------------------------------------------------------------------------

def curve_cdf(encoded_curve, bins=BINS):
    clipped = np.clip(encoded_curve, 0.0, 1.0)
    hist, _ = np.histogram(clipped, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total > 0:
        return np.cumsum(hist.astype(np.float64)) / total
    return np.zeros(bins, dtype=np.float64)


def curve_histogram(encoded_curve, bins=BINS):
    clipped = np.clip(encoded_curve, 0.0, 1.0)
    hist, _ = np.histogram(clipped, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def normalize_cdf(cdf):
    """Scale CDF to span [0, 1] so shape comparison ignores overall brightness."""
    lo, hi = cdf[0], cdf[-1]
    span = hi - lo
    if span < 1e-10:
        return np.zeros_like(cdf)
    return (cdf - lo) / span


def pearson_correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def anchor_similarity(footage_anchors, reference_anchors):
    """Similarity based on weighted anchor-point distance.

    Black level weighted 3x (scene-independent), mid-grey 1x, shoulder 1x.
    Returns 0..1 where 1 = perfect match.
    """
    weights = np.array([3.0, 1.0, 1.0])
    weighted_dist = np.sum(weights * np.abs(footage_anchors - reference_anchors)) / np.sum(weights)
    return float(np.exp(-10.0 * weighted_dist))


# ---------------------------------------------------------------------------
# Rank fusion
# ---------------------------------------------------------------------------

def assign_ranks(values, higher_is_better=True):
    """Return 1-based ranks for a list of values. Rank 1 = best."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=higher_is_better)
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


def compute_rank_fusion(results):
    """Compute independent ranks for each metric, then average.

    Mutates each result dict in-place: adds rank_hist, rank_cdf,
    rank_anchor, avg_rank.
    """
    hist_vals = [r["hist_corr"] for r in results]
    cdf_vals = [r["cdf_corr"] for r in results]
    anchor_vals = [r["anchor_sim"] for r in results]

    hist_ranks = assign_ranks(hist_vals, higher_is_better=True)
    cdf_ranks = assign_ranks(cdf_vals, higher_is_better=True)
    anchor_ranks = assign_ranks(anchor_vals, higher_is_better=True)

    for i, r in enumerate(results):
        r["rank_hist"] = hist_ranks[i]
        r["rank_cdf"] = cdf_ranks[i]
        r["rank_anchor"] = anchor_ranks[i]
        r["avg_rank"] = (hist_ranks[i] + cdf_ranks[i] + anchor_ranks[i]) / 3.0


# ---------------------------------------------------------------------------
# Visualization — side-by-side histogram + CDF
# ---------------------------------------------------------------------------

def _plot_panel(ax, bin_centers, foot_data, results, data_key, label_foot,
                ylabel, legend_loc, top_colors):
    """Shared logic for the histogram (left) and CDF (right) panels."""
    # Footage — filled blue
    ax.fill_between(bin_centers, foot_data, alpha=0.35, color="#4488cc",
                    label=label_foot, zorder=2)

    # Non-top curves (thin dashed grey, no legend)
    for i in range(len(results) - 1, 2, -1):
        ax.plot(bin_centers, results[i][data_key], color="#aaaaaa",
                linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

    # Top 3 — solid thick coloured lines
    for i in range(min(3, len(results)) - 1, -1, -1):
        r = results[i]
        ax.plot(bin_centers, r[data_key], color=top_colors[i], linewidth=2.0,
                label=f"{r['idt']}  (avg rank {r['avg_rank']:.1f})",
                zorder=3 + (3 - i))

    # Ranks 4-5 in legend only
    for i in range(3, min(5, len(results))):
        r = results[i]
        ax.plot([], [], color="#aaaaaa", linewidth=0.7, linestyle="--",
                label=f"{r['idt']}  (avg rank {r['avg_rank']:.1f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Code Value")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc=legend_loc)


def build_results_figure(foot_hist, foot_cdf, results, output_path, source_name):
    """Side-by-side: left = histogram comparison, right = CDF comparison."""
    bin_centers = np.linspace(0.0, 1.0, BINS, endpoint=False)
    bin_centers += (bin_centers[1] - bin_centers[0]) / 2.0

    top_colors = ["#dd4422", "#22aa88", "#aa44cc"]

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(22, 7))
    fig.suptitle(f"CurveMatcher — {source_name}", fontsize=14, fontweight="bold")

    # Left panel — histogram comparison
    _plot_panel(ax_hist, bin_centers, foot_hist, results, "ref_hist",
                "Footage histogram", "Density", "upper right", top_colors)
    ax_hist.set_ylim(bottom=0.0)
    ax_hist.set_title("Histogram Comparison", fontsize=11)

    # Right panel — CDF comparison
    _plot_panel(ax_cdf, bin_centers, foot_cdf, results, "ref_cdf",
                "Footage CDF", "Cumulative Probability", "lower right", top_colors)
    ax_cdf.set_ylim(0.0, 1.05)
    ax_cdf.set_title("CDF Comparison", fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CurveMatcher — identify camera encoding by matching "
                    "footage against known log transfer functions "
                    "(histogram + CDF + anchor-point rank fusion)")
    parser.add_argument("video",
                        help="Path to source video or still image")
    parser.add_argument("--config", default=None,
                        help="Path to OCIO config (.ocio). "
                             "Default: built-in ACES Studio Config v4.0.0")
    parser.add_argument("--samples", type=int, default=7,
                        help="Number of frames to sample from video (default: 7)")
    parser.add_argument("--output", default="output/curve_match",
                        help="Output directory (default: output/curve_match)")
    args = parser.parse_args()

    # --- OCIO config ---
    if args.config:
        config = OCIO.Config.CreateFromFile(args.config)
        print(f"Using OCIO config: {args.config}")
    else:
        config = OCIO.Config.CreateFromBuiltinConfig(
            "studio-config-v4.0.0_aces-v2.0_ocio-v2.5")
        print("Using built-in ACES Studio Config v4.0.0")

    # --- Load footage ---
    src = args.video
    if src.lower().endswith(STILL_IMAGE_EXTENSIONS):
        print(f"Still image detected: {os.path.basename(src)}")
        frames = load_image(src)
    else:
        frames = extract_frames(src, args.samples)

    source_name = os.path.basename(src)
    print(f"Sampled {len(frames)} frame(s)\n")

    # --- Footage descriptors (computed once) ---
    foot_hist = footage_histogram(frames)
    foot_cdf = footage_cdf_averaged(frames)
    foot_cdf_norm = normalize_cdf(foot_cdf)

    bin_edges = np.linspace(0.0, 1.0, BINS + 1)
    foot_anchors = detect_footage_anchors(foot_cdf, bin_edges)

    # --- Score each candidate IDT ---
    results = []
    for idt in IDTs:
        try:
            curve = generate_encoding_curve(config, idt)
            ref_anchors = generate_reference_anchors(config, idt)

            c_cdf = curve_cdf(curve)
            c_cdf_norm = normalize_cdf(c_cdf)
            c_hist = curve_histogram(curve)

            cdf_corr = pearson_correlation(foot_cdf_norm, c_cdf_norm)
            anch_sim = anchor_similarity(foot_anchors, ref_anchors)
            hist_corr = pearson_correlation(foot_hist, c_hist)

            results.append({
                "idt": idt,
                "hist_corr": hist_corr,
                "cdf_corr": cdf_corr,
                "anchor_sim": anch_sim,
                "ref_cdf": c_cdf,
                "ref_hist": c_hist,
                "ref_anchors": ref_anchors,
            })
        except Exception as e:
            print(f"  Skipping {idt}: {e}")

    # --- Rank fusion ---
    compute_rank_fusion(results)
    results.sort(key=lambda r: r["avg_rank"])

    # --- Console table ---
    print(f"CurveMatcher results for: {source_name}")
    print(f"  Footage anchors — black: {foot_anchors[0]:.3f}  "
          f"mid: {foot_anchors[1]:.3f}  shoulder: {foot_anchors[2]:.3f}\n")

    hdr = (f"{'Rank':<6}{'IDT':<40}"
           f"{'Hist r':>8} {'#':>3}  "
           f"{'CDF r':>8} {'#':>3}  "
           f"{'Anchor':>8} {'#':>3}  "
           f"{'Avg Rank':>9}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results, 1):
        print(f"{i:<6}{r['idt']:<40}"
              f"{r['hist_corr']:>8.4f} {r['rank_hist']:>3d}  "
              f"{r['cdf_corr']:>8.4f} {r['rank_cdf']:>3d}  "
              f"{r['anchor_sim']:>8.4f} {r['rank_anchor']:>3d}  "
              f"{r['avg_rank']:>9.2f}")

    if results:
        best = results[0]
        print(f"\nBest match: {best['idt']}  "
              f"(avg rank = {best['avg_rank']:.2f})")
        print(f"  Reference anchors — black: {best['ref_anchors'][0]:.3f}  "
              f"mid: {best['ref_anchors'][1]:.3f}  "
              f"shoulder: {best['ref_anchors'][2]:.3f}")

    # --- Visual output ---
    stem = os.path.splitext(source_name)[0]
    plot_path = os.path.join(args.output, f"{stem}_curve_match.png")
    build_results_figure(foot_hist, foot_cdf, results, plot_path, source_name)


if __name__ == "__main__":
    main()
