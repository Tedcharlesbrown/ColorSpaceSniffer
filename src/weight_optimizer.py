#!/usr/bin/env python3
"""
weight_optimizer.py — brute-force weight search for ColorSpaceSniffer.

Phase 1 (slow): Cache raw metric values for all test files × IDTs (84 evaluations).
Phase 2 (fast): Vectorized random weight search over N combos using cached data.

Usage:
    source .venv/bin/activate
    python3 src/weight_optimizer.py [--combos 50000]
"""

import sys
import os
import csv
import math
import time
import argparse
import numpy as np
import PyOpenColorIO as OCIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing import load_image, OCIO_CST
from metrics import (
    CheckNeutralVariance,
    ManyFrameChannelCorrelation,
    CheckGamutClipping,
    LocalContrastRatio,
    SceneLinearRangeCheck,
    HistogramEntropy,
    BandingDetection,
    ShadowNoiseAmplification,
    GreyWorldDeviation,
    IterativeDecodeDivergence,
    ShadowArtifactMonotonicity,
    SpectralLocusViolation,
    RoundTripPSNR,
)

# ── Constants ────────────────────────────────────────────────────────────────

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'examples', 'images')

GROUND_TRUTH = {
    'DXR_CAM01_SLog3.tiff': [
        'S-Log3 S-Gamut3', 'S-Log3 S-Gamut3.Cine',
        'S-Log3 Venice S-Gamut3', 'S-Log3 Venice S-Gamut3.Cine'],
    'DXR_CAM09_SLog3.tiff': [
        'S-Log3 S-Gamut3', 'S-Log3 S-Gamut3.Cine',
        'S-Log3 Venice S-Gamut3', 'S-Log3 Venice S-Gamut3.Cine'],
    'DPTest_Cam01_VLog.tiff':     ['V-Log V-Gamut'],
    'DPTest_Cam03_VLog.tiff':     ['V-Log V-Gamut'],
    'DPTest_Cam09Fish_VLog.tiff': ['V-Log V-Gamut'],
    'DPTest_Cam09Rect_VLog.tiff': ['V-Log V-Gamut'],
}

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

ODT = "P3-D65 - Display"

# ── Feature definitions ──────────────────────────────────────────────────────
# Each entry: (name, search_range_min, search_range_max)
# Naming convention:
#   Positive weight → higher feature value = worse (penalty)
#   Negative weight → higher feature value = better (reward)
#
# Pre-processing applied before the weight multiply:
#   lcr_low  = max(0, 1.5 - lcr)      near-identity decode
#   lcr_high = max(0, lcr - 3.5)      extreme expansion
#   shadow   = max(0, shadow - 1.0)   threshold out normal amplification
#   divergence = log1p(min(raw, 1e6)) compress the divergence ratio

FEATURE_DEFS = [
    # name          lo      hi
    ('nv',        0.0,   5.0),   # NeutralVariance          (higher = worse)
    ('cc',        0.0,   3.0),   # ChannelCorrelation       (higher = worse)
    ('clip',      0.0,   2.0),   # GamutClipping            (higher = worse)
    ('lcr_low',   0.0,   4.0),   # LCR near-identity        (higher = worse)
    ('lcr_high',  0.0,   2.0),   # LCR extreme expansion    (higher = worse)
    ('slr',       0.0,   5.0),   # SceneLinearRange         (higher = worse)
    ('entropy',  -1.0,   0.5),   # HistogramEntropy         (higher = BETTER → neg weight)
    ('banding',   0.0,   2.0),   # BandingDetection         (higher = worse)
    ('shadow',    0.0,   2.0),   # ShadowNoise (thresholded)(higher = worse)
    ('grey',      0.0,   2.0),   # GreyWorldDeviation       (higher = worse)
    ('divergence',-2.0,  0.5),   # IDD log1p                (higher = BETTER → neg weight)
    ('sam',       -3.0,  1.0),   # ShadowArtifactMonoton.   (higher = BETTER → neg weight)
    ('slv',       -3.0,  5.0),   # SpectralLocusViolation   (direction uncertain)
]

FEAT_NAMES = [d[0] for d in FEATURE_DEFS]
FEAT_LO    = np.array([d[1] for d in FEATURE_DEFS], dtype=np.float64)
FEAT_HI    = np.array([d[2] for d in FEATURE_DEFS], dtype=np.float64)
N_FEATURES = len(FEATURE_DEFS)

# Current weights (for baseline comparison)
CURRENT_WEIGHTS = np.array([
    2.0,   # nv
    1.5,   # cc
    1.0,   # clip
    2.0,   # lcr_low
    0.5,   # lcr_high
    3.0,   # slr
   -0.3,   # entropy
    1.5,   # banding
    1.0,   # shadow
    1.0,   # grey
   -0.3,   # divergence (log1p)
   -2.5,   # sam
    0.0,   # slv
], dtype=np.float64)

# ── Metric computation ───────────────────────────────────────────────────────

def compute_raw_metrics(frames, config, idt, odt):
    """Compute all raw metric values for one (frames, IDT) pair. Returns dict."""
    transformed = OCIO_CST(frames, config, idt, odt)

    clip_ratios = CheckGamutClipping(transformed)
    raw_clip = sum(max(0.0, r - 0.05) for r in clip_ratios)

    return {
        'nv':        CheckNeutralVariance(transformed),
        'cc':        ManyFrameChannelCorrelation(transformed),
        'clip':      raw_clip,
        'lcr':       LocalContrastRatio(frames, transformed),
        'slr':       SceneLinearRangeCheck(frames, config, idt),
        'entropy':   HistogramEntropy(transformed),
        'banding':   BandingDetection(transformed),
        'shadow':    ShadowNoiseAmplification(frames, transformed),
        'grey':      GreyWorldDeviation(transformed),
        'divergence':IterativeDecodeDivergence(frames, config, idt),
        'sam':       ShadowArtifactMonotonicity(frames, config, idt),
        'slv':       SpectralLocusViolation(frames, config, idt),
        'psnr':      RoundTripPSNR(frames, transformed, config, idt, odt),
    }


def _safe(v, fallback=0.0):
    return fallback if (math.isnan(v) or math.isinf(v)) else v


def raw_to_feature_vec(raw):
    """Apply non-linear pre-processing and return feature vector + PSNR."""
    lcr       = _safe(raw['lcr'], fallback=4.0)
    shadow    = _safe(raw['shadow'], fallback=1.0)
    divergence= _safe(raw['divergence'], fallback=0.0)
    slv       = _safe(raw.get('slv', 0.0), fallback=0.0)

    lcr_low   = max(0.0, 1.5 - lcr)
    lcr_high  = max(0.0, lcr - 3.5)
    shadow_t  = max(0.0, shadow - 1.0)
    div_log   = math.log1p(min(divergence, 1e6))

    vec = np.array([
        _safe(raw['nv']),
        _safe(raw['cc']),
        _safe(raw['clip']),
        lcr_low,
        lcr_high,
        _safe(raw['slr']),
        _safe(raw['entropy']),
        _safe(raw['banding']),
        shadow_t,
        _safe(raw['grey']),
        div_log,
        _safe(raw['sam']),
        slv,
    ], dtype=np.float64)

    psnr = _safe(raw['psnr'], fallback=48.0)
    return vec, psnr


# ── Phase 1: Cache ───────────────────────────────────────────────────────────

def build_cache(config):
    """Evaluate all test files × IDTs and cache results.

    Returns:
        feature_matrix : np.ndarray (n_files, n_idts, n_features)
        psnr_matrix    : np.ndarray (n_files, n_idts)
        file_names     : list[str]
        correct_indices: list[list[int]]  correct IDT indices per file
        raw_cache      : dict for optional CSV dump
    """
    file_names = list(GROUND_TRUTH.keys())
    n_files    = len(file_names)
    n_idts     = len(IDTs)

    feature_matrix = np.zeros((n_files, n_idts, N_FEATURES), dtype=np.float64)
    psnr_matrix    = np.zeros((n_files, n_idts),             dtype=np.float64)
    raw_cache      = {}   # {filename: {idt: raw_dict}}

    correct_indices = []
    for fname in file_names:
        correct_indices.append([IDTs.index(idt)
                                 for idt in GROUND_TRUTH[fname] if idt in IDTs])

    total = n_files * n_idts
    done  = 0
    phase1_start = time.time()

    for fi, fname in enumerate(file_names):
        fpath = os.path.join(IMAGES_DIR, fname)
        print(f"\n[File {fi+1}/{n_files}] {fname}")
        frames = load_image(fpath)
        raw_cache[fname] = {}

        for ii, idt in enumerate(IDTs):
            done += 1
            print(f"  [{done:2d}/{total}] {idt:<40}", end='', flush=True)
            t0 = time.time()
            try:
                raw = compute_raw_metrics(frames, config, idt, ODT)
                feat, psnr = raw_to_feature_vec(raw)
                feature_matrix[fi, ii, :] = feat
                psnr_matrix[fi, ii]        = psnr
                raw_cache[fname][idt]      = raw
                print(f" {time.time()-t0:.1f}s")
            except Exception as e:
                print(f" ERROR: {e}")
                psnr_matrix[fi, ii] = 48.0   # neutral PSNR divisor

    elapsed = time.time() - phase1_start
    print(f"\nPhase 1 complete: {total} evaluations in {elapsed:.1f}s "
          f"({elapsed/total:.1f}s avg)\n")

    return feature_matrix, psnr_matrix, file_names, correct_indices, raw_cache


# ── Phase 2: Weight search ───────────────────────────────────────────────────

def score_weight_combos(feature_matrix, psnr_matrix, correct_indices,
                        weights_matrix):
    """Vectorised correctness scoring for many weight combos.

    Args:
        feature_matrix : (n_files, n_idts, n_features)
        psnr_matrix    : (n_files, n_idts)
        correct_indices: list[list[int]]
        weights_matrix : (n_combos, n_features)

    Returns:
        points  : np.ndarray (n_combos,)  total points  (max = n_files × 3)
        top3    : np.ndarray (n_combos,)  #1 hits       (max = n_files)
    """
    n_files, n_idts, _ = feature_matrix.shape
    n_combos = weights_matrix.shape[0]

    total_points = np.zeros(n_combos, dtype=np.float32)
    total_top1   = np.zeros(n_combos, dtype=np.float32)

    for fi in range(n_files):
        F = feature_matrix[fi]          # (n_idts, n_features)
        P = psnr_matrix[fi]             # (n_idts,)

        # (n_idts, n_combos)
        raw_scores   = F @ weights_matrix.T
        psnr_norm    = np.maximum(P / 48.0, 0.1)[:, None]   # (n_idts, 1)
        final_scores = raw_scores / psnr_norm                # (n_idts, n_combos)

        # rank_of_idt[i, j] = rank of IDT i in combo j  (0 = best / lowest score)
        rank_of_idt = np.argsort(np.argsort(final_scores, axis=0), axis=0)

        ci = correct_indices[fi]                              # list of ints
        # best rank among all acceptable IDTs for this file, per combo
        best_rank = np.min(rank_of_idt[ci, :], axis=0)       # (n_combos,)

        total_points += np.where(best_rank == 0, 3,
                        np.where(best_rank <= 2, 1, 0))
        total_top1   += (best_rank == 0).astype(np.float32)

    return total_points, total_top1


def per_file_ranks(feature_matrix, psnr_matrix, correct_indices, weights):
    """Return the best correct IDT rank for each file under a given weight vector."""
    n_files = feature_matrix.shape[0]
    result  = []
    for fi in range(n_files):
        F = feature_matrix[fi]
        P = psnr_matrix[fi]
        scores = (F @ weights) / np.maximum(P / 48.0, 0.1)
        ranks  = np.argsort(np.argsort(scores))
        ci     = correct_indices[fi]
        best   = int(np.min(ranks[ci])) + 1   # 1-based
        # also grab the name of whichever correct IDT ranked best
        best_idt_idx = ci[int(np.argmin(ranks[ci]))]
        result.append((best, IDTs[best_idt_idx]))
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ColorSpaceSniffer weight optimizer')
    parser.add_argument('--combos', type=int, default=50000,
                        help='Number of random weight combos to try (default 50000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', default='/tmp/weight_optimization_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    config = OCIO.Config.CreateFromBuiltinConfig(
        'studio-config-v4.0.0_aces-v2.0_ocio-v2.5')
    print("Using built-in ACES Studio Config v4.0.0\n")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Caching raw metrics")
    print(f"  {len(GROUND_TRUTH)} files × {len(IDTs)} IDTs = "
          f"{len(GROUND_TRUTH)*len(IDTs)} evaluations")
    print("=" * 60)

    feature_matrix, psnr_matrix, file_names, correct_indices, raw_cache = \
        build_cache(config)

    # Print raw feature values for inspection
    print("\n── Cached feature values (pre-processed) ──────────────────")
    header = f"{'IDT':<40}" + " ".join(f"{n:>8}" for n in FEAT_NAMES)
    print(header)
    for fi, fname in enumerate(file_names):
        print(f"\n  {fname}")
        for ii, idt in enumerate(IDTs):
            row = f"  {idt:<38}" + " ".join(
                f"{feature_matrix[fi, ii, k]:>8.3f}" for k in range(N_FEATURES))
            ci  = correct_indices[fi]
            tag = " ← CORRECT" if ii in ci else ""
            print(row + tag)

    # Baseline score with current weights
    base_pts, base_top1 = score_weight_combos(
        feature_matrix, psnr_matrix, correct_indices,
        CURRENT_WEIGHTS[None, :])
    print(f"\n── Baseline (current weights): "
          f"{int(base_pts[0])}/18 pts, {int(base_top1[0])}/6 top-1 hits ──")
    for (rank, best_idt), fname in zip(
            per_file_ranks(feature_matrix, psnr_matrix,
                           correct_indices, CURRENT_WEIGHTS),
            file_names):
        tag = "✓" if rank == 1 else ("~" if rank <= 3 else "✗")
        print(f"  {tag} {fname:<40} correct IDT rank #{rank}  ({best_idt})")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    n_combos = args.combos
    print(f"\n{'='*60}")
    print(f"PHASE 2: Random weight search ({n_combos:,} combos)")
    print("=" * 60)

    rng = np.random.default_rng(args.seed)

    # Sample uniformly across each dimension's range
    weights_matrix = (rng.random((n_combos, N_FEATURES))
                      * (FEAT_HI - FEAT_LO) + FEAT_LO).astype(np.float64)

    # Always include the current weights as combo 0
    weights_matrix[0] = CURRENT_WEIGHTS

    t0 = time.time()

    # Evaluate in batches of 5000 to show progress
    batch_size = 5000
    all_pts  = np.zeros(n_combos, dtype=np.float32)
    all_top1 = np.zeros(n_combos, dtype=np.float32)

    for start in range(0, n_combos, batch_size):
        end = min(start + batch_size, n_combos)
        pts, top1 = score_weight_combos(
            feature_matrix, psnr_matrix, correct_indices,
            weights_matrix[start:end])
        all_pts[start:end]  = pts
        all_top1[start:end] = top1
        elapsed = time.time() - t0
        rate    = (end) / elapsed if elapsed > 0 else 0
        print(f"  {end:>6,}/{n_combos:,}  "
              f"best so far: {int(all_pts[:end].max())}/18 pts  "
              f"({rate:.0f} combos/s)", end='\r')

    elapsed = time.time() - t0
    print(f"\nPhase 2 complete: {n_combos:,} combos in {elapsed:.1f}s "
          f"({n_combos/elapsed:.0f} combos/s)")

    # ── Results ───────────────────────────────────────────────────────────────
    # Sort by points descending, then top1 descending
    order = np.lexsort((-all_top1, -all_pts))

    print(f"\n{'='*60}")
    print("TOP 20 WEIGHT COMBINATIONS")
    print("=" * 60)
    col_w = 7
    header = (f"{'Rank':>4}  {'Pts':>4}  {'#1':>3}  " +
              "  ".join(f"{n:>{col_w}}" for n in FEAT_NAMES))
    print(header)
    print("-" * len(header))

    top_results = []
    for rank_i, idx in enumerate(order[:20]):
        w  = weights_matrix[idx]
        pts  = int(all_pts[idx])
        top1 = int(all_top1[idx])
        row = (f"{rank_i+1:>4}  {pts:>4}  {top1:>3}  " +
               "  ".join(f"{w[k]:>{col_w}.3f}" for k in range(N_FEATURES)))
        tag = "  ← current" if idx == 0 else ""
        print(row + tag)
        top_results.append({'rank': rank_i+1, 'points': pts, 'top1': top1,
                             **{FEAT_NAMES[k]: float(w[k]) for k in range(N_FEATURES)}})

    # ── Best weight detail ────────────────────────────────────────────────────
    best_idx = order[0]
    best_w   = weights_matrix[best_idx]
    print(f"\n{'='*60}")
    print("BEST WEIGHTS — per-file breakdown")
    print("=" * 60)
    for (rank, best_idt), fname in zip(
            per_file_ranks(feature_matrix, psnr_matrix,
                           correct_indices, best_w),
            file_names):
        tag = "✓" if rank == 1 else ("~" if rank <= 3 else "✗")
        print(f"  {tag} {fname:<40} correct IDT rank #{rank}  ({best_idt})")

    print(f"\nBest weight vector (copy-paste ready):")
    for name, val in zip(FEAT_NAMES, best_w):
        print(f"    {name:<12} = {val:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    # Save top 2000 results
    top_n    = min(2000, n_combos)
    top_idxs = order[:top_n]

    fieldnames = ['rank', 'points', 'top1'] + FEAT_NAMES
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ri, idx in enumerate(top_idxs):
            w = weights_matrix[idx]
            writer.writerow({
                'rank':   ri + 1,
                'points': int(all_pts[idx]),
                'top1':   int(all_top1[idx]),
                **{FEAT_NAMES[k]: round(float(w[k]), 5) for k in range(N_FEATURES)},
            })
    print(f"\nTop {top_n} results saved to: {args.output}")


if __name__ == '__main__':
    main()
