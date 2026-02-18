"""
Gamut Wall Test — Detect whether footage is log/wide-gamut or display-referred.

Maps raw pixel values to CIE xy chromaticity and analyzes the convex hull shape.
Display-referred footage (baked LUT → Rec.709/sRGB) shows hard geometric edges
aligned with the Rec.709 triangle. Log/wide-gamut footage shows an amorphous blob.

Usage:
    python gamut_wall.py <image_path> [--plot]
"""

import numpy as np
import argparse
import os
import sys

# Rec.709 / sRGB primaries in CIE xy
REC709_PRIMARIES = np.array([
    [0.64, 0.33],   # R
    [0.30, 0.60],   # G
    [0.15, 0.06],   # B
])

# Rec.2020 primaries in CIE xy
REC2020_PRIMARIES = np.array([
    [0.708, 0.292],  # R
    [0.170, 0.797],  # G
    [0.131, 0.046],  # B
])

# P3-D65 primaries
P3D65_PRIMARIES = np.array([
    [0.680, 0.320],  # R
    [0.265, 0.690],  # G
    [0.150, 0.060],  # B
])

# sRGB linearization (inverse OETF)
def srgb_to_linear(v):
    """Convert sRGB-encoded values to linear light."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

# BT.709 matrix: linear RGB → CIE XYZ (D65)
# Standard sRGB/Rec.709 to XYZ matrix
RGB_TO_XYZ_709 = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def rgb_to_xy_chromaticity(rgb_pixels):
    """
    Convert RGB pixels (N×3, 0-1 range) to CIE xy chromaticity.
    Assumes pixels are in some RGB space — we don't linearize here
    because we're looking at the RAW encoded values' chromaticity distribution.
    The gamut wall pattern exists in the encoded domain.
    """
    # Treat RGB values as proportional to tristimulus-like quantities
    # For gamut wall detection, we care about the SHAPE of the distribution,
    # not absolute colorimetric accuracy. Using raw RGB ratios works because:
    # - If footage is Rec.709, the hard clipping at primary boundaries is visible
    # - If footage is wide gamut, the distribution is smoother and more spread
    
    # Simple chromaticity: r = R/(R+G+B), g = G/(R+G+B)
    # This is rg chromaticity, not CIE xy, but it reveals the same gamut wall pattern
    total = rgb_pixels[:, 0] + rgb_pixels[:, 1] + rgb_pixels[:, 2]
    
    # Avoid division by zero (pure black pixels)
    valid = total > 1e-6
    r = np.zeros(len(rgb_pixels))
    g = np.zeros(len(rgb_pixels))
    r[valid] = rgb_pixels[valid, 0] / total[valid]
    g[valid] = rgb_pixels[valid, 1] / total[valid]
    
    return r[valid], g[valid]


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """Distance from point (px, py) to line segment (x1,y1)-(x2,y2)."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto line, clamped to segment
    t = np.clip(((px - x1) * dx + (py - y1) * dy) / length_sq, 0.0, 1.0)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)


def gamut_wall_score(r, g, primaries, threshold=0.005):
    """
    Measure what fraction of edge pixels cluster along the gamut triangle edges.
    
    Returns:
        wall_density: fraction of outermost pixels that lie near a triangle edge
        edge_counts: how many pixels cluster near each of the 3 edges
    """
    # Find the convex hull boundary pixels — use the outermost 5% by distance from centroid
    centroid_r = np.mean(r)
    centroid_g = np.mean(g)
    dist_from_center = np.sqrt((r - centroid_r)**2 + (g - centroid_g)**2)
    
    outer_threshold = np.percentile(dist_from_center, 95)
    outer_mask = dist_from_center >= outer_threshold
    outer_r = r[outer_mask]
    outer_g = g[outer_mask]
    
    # Convert primaries to rg chromaticity for comparison
    # rg chromaticity of primaries: r_i = R_i / (R_i + G_i + B_i) etc.
    # For Rec.709 primaries in CIE xy, approximate rg chromaticity:
    # Pure red (1,0,0) → r=1, g=0
    # Pure green (0,1,0) → r=0, g=1  
    # Pure blue (0,0,1) → r=0, g=0
    rg_primaries = np.array([
        [1.0, 0.0],  # Pure R
        [0.0, 1.0],  # Pure G
        [0.0, 0.0],  # Pure B
    ])
    
    # Three edges of the triangle
    edges = [
        (rg_primaries[0], rg_primaries[1]),  # R-G edge
        (rg_primaries[1], rg_primaries[2]),  # G-B edge
        (rg_primaries[2], rg_primaries[0]),  # B-R edge
    ]
    
    near_edge = np.zeros(len(outer_r), dtype=bool)
    edge_counts = []
    
    for (p1, p2) in edges:
        dists = np.array([
            point_to_line_distance(outer_r[i], outer_g[i], p1[0], p1[1], p2[0], p2[1])
            for i in range(len(outer_r))
        ])
        close = dists < threshold
        edge_counts.append(np.sum(close))
        near_edge |= close
    
    wall_density = np.sum(near_edge) / max(len(outer_r), 1)
    return wall_density, edge_counts


def analyze_chromaticity_spread(r, g):
    """
    Measure the overall shape/spread of the chromaticity distribution.
    
    Log/wide-gamut: concentrated blob, moderate spread
    Display-referred: wider spread with hard edges
    """
    # Standard deviations
    std_r = np.std(r)
    std_g = np.std(g)
    
    # Kurtosis (Fisher's definition) — numpy only
    def _kurtosis(arr):
        m = np.mean(arr)
        s = np.std(arr)
        if s < 1e-10:
            return 0.0
        return np.mean(((arr - m) / s) ** 4) - 3.0
    
    kurt_r = _kurtosis(r)
    kurt_g = _kurtosis(g)
    
    # Approximate convex hull area using 2D histogram extent
    # (Avoids scipy dependency — good enough for our diagnostic)
    # Use the area of the bounding box of the 1st-99th percentile
    r_range = np.percentile(r, 99) - np.percentile(r, 1)
    g_range = np.percentile(g, 99) - np.percentile(g, 1)
    hull_area = r_range * g_range  # Approximate
    
    return {
        'std_r': std_r,
        'std_g': std_g,
        'kurtosis_r': kurt_r,
        'kurtosis_g': kurt_g,
        'hull_area': hull_area,
    }


def detect_hard_edges(r, g, n_bins=200):
    """
    Detect hard edges in the chromaticity distribution by looking for
    sharp density dropoffs — the "wall" signature.
    
    Returns a sharpness score: higher = more hard edges = more likely display-referred.
    """
    # 2D histogram of chromaticity
    hist, xedges, yedges = np.histogram2d(r, g, bins=n_bins, range=[[0, 1], [0, 1]])
    
    # Compute gradient magnitude of the 2D histogram
    # Sharp transitions (occupied → empty) indicate gamut walls
    grad_x = np.diff(hist, axis=0)
    grad_y = np.diff(hist, axis=1)
    
    # Trim to same size
    min_rows = min(grad_x.shape[0], grad_y.shape[0])
    min_cols = min(grad_x.shape[1], grad_y.shape[1])
    grad_x = grad_x[:min_rows, :min_cols]
    grad_y = grad_y[:min_rows, :min_cols]
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # The "sharpness" score: ratio of max gradient to mean gradient
    # Hard walls produce extreme gradients; smooth blobs don't
    mean_grad = np.mean(grad_mag[grad_mag > 0]) if np.any(grad_mag > 0) else 1e-6
    max_grad = np.max(grad_mag)
    p99_grad = np.percentile(grad_mag[grad_mag > 0], 99) if np.any(grad_mag > 0) else 0
    
    # Count bins with very sharp transitions (above 10x mean)
    sharp_bin_count = np.sum(grad_mag > 10 * mean_grad)
    
    return {
        'max_gradient': max_grad,
        'p99_gradient': p99_grad,
        'mean_gradient': mean_grad,
        'sharpness_ratio': max_grad / mean_grad if mean_grad > 0 else 0,
        'sharp_bin_count': sharp_bin_count,
    }


def gamut_wall_test(image_frames, verbose=True, output_path=None, image_name=None):
    """
    Main entry point: analyze frames and determine if footage is log or display-referred.
    
    Args:
        image_frames: list of numpy arrays (H, W, 3), normalized 0-1 float
        verbose: print diagnostic info
        output_path: directory to save chromaticity plot (optional)
        image_name: base name for plot file (optional)
        
    Returns:
        dict with:
            'is_display_referred': bool (True if likely display-referred)
            'confidence': float 0-1
            'wall_density': float
            'edge_sharpness': dict
            'chromaticity_stats': dict
    """
    # Collect all pixels from all frames
    all_pixels = []
    for frame in image_frames:
        h, w = frame.shape[:2]
        pixels = frame.reshape(-1, 3)
        # Subsample for speed — 500k pixels is plenty for statistics
        if len(pixels) > 500000:
            idx = np.random.RandomState(42).choice(len(pixels), 500000, replace=False)
            pixels = pixels[idx]
        all_pixels.append(pixels)
    
    pixels = np.vstack(all_pixels)
    
    # --- Luminance analysis (strongest display-referred signal) ---
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    mean_luma = np.mean(luma)
    median_luma = np.median(luma)
    clipped_white_pct = np.sum(luma >= 0.99) / len(luma)
    clipped_black_pct = np.sum(luma <= 0.001) / len(luma)
    
    luma_std = np.std(luma)
    if luma_std > 1e-10:
        luma_m = np.mean(luma)
        luma_skew = np.mean(((luma - luma_m) / luma_std) ** 3)
        luma_kurt = np.mean(((luma - luma_m) / luma_std) ** 4) - 3.0
    else:
        luma_skew = 0.0
        luma_kurt = 0.0
    
    # Convert to rg chromaticity
    r, g = rgb_to_xy_chromaticity(pixels)
    
    if verbose:
        print(f"\n--- Gamut Wall Test ---")
        print(f"  Analyzing {len(r):,} pixels")
        print(f"  Luminance: mean={mean_luma:.4f}  median={median_luma:.4f}  skew={luma_skew:.3f}  kurt={luma_kurt:.3f}")
        print(f"  Clipped whites (≥0.99): {100*clipped_white_pct:.1f}%  |  True blacks (≤0.001): {100*clipped_black_pct:.1f}%")
    
    # 1. Check for hard edges in chromaticity histogram
    edge_info = detect_hard_edges(r, g)
    
    # 2. Measure wall density along known gamut boundaries
    wall_density, edge_counts = gamut_wall_score(r, g, REC709_PRIMARIES)
    
    # 3. Analyze overall chromaticity distribution shape
    chroma_stats = analyze_chromaticity_spread(r, g)
    
    if verbose:
        print(f"  Chromaticity spread: std_r={chroma_stats['std_r']:.4f}, std_g={chroma_stats['std_g']:.4f}")
        print(f"  Kurtosis: r={chroma_stats['kurtosis_r']:.2f}, g={chroma_stats['kurtosis_g']:.2f}")
        print(f"  Convex hull area: {chroma_stats['hull_area']:.6f}")
        print(f"  Edge sharpness ratio: {edge_info['sharpness_ratio']:.1f}")
        print(f"  Sharp gradient bins: {edge_info['sharp_bin_count']}")
        print(f"  Wall density (outer 5% near rg triangle): {wall_density:.3f}")
        print(f"  Edge pixel counts [R-G, G-B, B-R]: {edge_counts}")
    
    # Decision logic (tuned on 18-file test matrix, 2026-02-17):
    #
    # Log/wide-gamut footage:
    #   - Concentrated chromaticity blob (hull_area < 0.01)
    #   - Very high kurtosis (>10) — peaked distribution
    #   - No wall density
    #
    # Display-referred footage:
    #   - Wider chromaticity spread (hull_area > 0.01)
    #   - Low kurtosis (<5) — flat/spread distribution
    #   - May or may not have wall density (depends on gamut: Rec.709 has walls, Rec.2020 may not)
    #
    # Key insight: rg chromaticity kurtosis is the strongest single signal.
    # Log encoding compresses chromaticity by lifting shadows toward neutral,
    # creating a peaked distribution. Display-referred footage preserves the
    # natural chromaticity spread.
    
    score = 0.0
    reasons = []
    avg_kurtosis = (chroma_stats['kurtosis_r'] + chroma_stats['kurtosis_g']) / 2.0
    
    # === LUMINANCE SIGNALS (strongest for display-referred detection) ===
    
    # Clipped whites — THE smoking gun. Log footage never clips at 1.0.
    # Log curves compress highlights, so max pixel values are well below 1.0.
    if clipped_white_pct > 0.05:  # >5% clipped whites
        score += 0.40
        reasons.append(f"Heavy highlight clipping ({100*clipped_white_pct:.1f}%)")
    elif clipped_white_pct > 0.01:  # >1%
        score += 0.25
        reasons.append(f"Moderate highlight clipping ({100*clipped_white_pct:.1f}%)")
    elif clipped_white_pct > 0.001:  # >0.1%
        score += 0.10
        reasons.append(f"Slight highlight clipping ({100*clipped_white_pct:.2f}%)")
    
    # High mean luminance — log footage is "flat and milky" (mean ~0.1-0.35)
    # Display-referred footage has higher mean (>0.40)
    if mean_luma > 0.50:
        score += 0.20
        reasons.append(f"High mean luminance ({mean_luma:.3f})")
    elif mean_luma > 0.40:
        score += 0.10
        reasons.append(f"Elevated mean luminance ({mean_luma:.3f})")
    
    # Negative luminance kurtosis — flat/uniform distribution typical of display
    if luma_kurt < -1.0:
        score += 0.15
        reasons.append(f"Negative luma kurtosis ({luma_kurt:.2f})")
    elif luma_kurt < -0.3:
        score += 0.05
    
    # === CHROMATICITY SIGNALS ===
    
    # LOW chromaticity kurtosis = spread out = display-referred
    if avg_kurtosis < 3.0:
        score += 0.15
    elif avg_kurtosis < 8.0:
        score += 0.05
    
    # Large hull area = wide chromaticity spread
    if chroma_stats['hull_area'] > 0.05:
        score += 0.15
    elif chroma_stats['hull_area'] > 0.015:
        score += 0.10
    elif chroma_stats['hull_area'] > 0.008:
        score += 0.05
    
    # Wall density — hard edges along known gamut triangles
    if wall_density > 0.3:
        score += 0.15
        reasons.append(f"Gamut wall detected (density={wall_density:.2f})")
    elif wall_density > 0.10:
        score += 0.05
    
    is_display = score >= 0.50
    
    if verbose:
        verdict = "🔴 DISPLAY-REFERRED" if is_display else "🟢 LOG / WIDE GAMUT"
        print(f"\n  Confidence score: {score:.2f}")
        print(f"  Verdict: {verdict}")
        if reasons:
            print(f"  Flags: {'; '.join(reasons)}")
        if is_display:
            print(f"  ⚠️  A viewing transform appears to be baked in.")
            print(f"     Standard log IDTs are unlikely to be correct.")
            print(f"     Consider inverse tone mapping (ITM) instead.")
    
    # Save chromaticity plot if output path provided
    if output_path and image_name:
        try:
            verdict_str = "DISPLAY" if is_display else "LOG"
            plot_chromaticity(pixels,
                              title=f"{image_name} — {verdict_str} (conf: {score:.2f})",
                              save_path=os.path.join(output_path, f"{image_name}_chromaticity.png"))
        except Exception as e:
            if verbose:
                print(f"  (Could not save chromaticity plot: {e})")
    
    return {
        'is_display_referred': is_display,
        'confidence': score,
        'wall_density': wall_density,
        'edge_counts': edge_counts,
        'edge_sharpness': edge_info,
        'chromaticity_stats': chroma_stats,
    }


# sRGB/Rec.709 to XYZ matrix (for CIE xy conversion)
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

# CIE 1931 spectral locus (sampled every 5nm, 380-700nm)
_SPECTRAL_LOCUS = np.array([
    [0.1741, 0.0050], [0.1740, 0.0050], [0.1738, 0.0049], [0.1736, 0.0049],
    [0.1733, 0.0048], [0.1726, 0.0048], [0.1714, 0.0051], [0.1689, 0.0069],
    [0.1644, 0.0109], [0.1566, 0.0177], [0.1440, 0.0297], [0.1241, 0.0578],
    [0.0913, 0.1327], [0.0687, 0.2007], [0.0454, 0.2950], [0.0235, 0.4127],
    [0.0082, 0.5384], [0.0039, 0.6548], [0.0139, 0.7502], [0.0389, 0.8120],
    [0.0743, 0.8338], [0.1142, 0.8262], [0.1547, 0.8059], [0.1929, 0.7816],
    [0.2296, 0.7543], [0.2658, 0.7243], [0.3016, 0.6923], [0.3373, 0.6589],
    [0.3731, 0.6245], [0.4087, 0.5896], [0.4441, 0.5547], [0.4788, 0.5202],
    [0.5125, 0.4866], [0.5448, 0.4544], [0.5752, 0.4242], [0.6029, 0.3965],
    [0.6270, 0.3725], [0.6482, 0.3514], [0.6658, 0.3340], [0.6801, 0.3197],
    [0.6915, 0.3083], [0.7006, 0.2993], [0.7079, 0.2920], [0.7140, 0.2859],
    [0.7190, 0.2809], [0.7230, 0.2770], [0.7260, 0.2740], [0.7283, 0.2717],
    [0.7300, 0.2700], [0.7311, 0.2689], [0.7320, 0.2680], [0.7327, 0.2673],
    [0.7334, 0.2666], [0.7340, 0.2660], [0.7344, 0.2656], [0.7346, 0.2654],
    [0.7347, 0.2653], [0.7347, 0.2653], [0.7347, 0.2653], [0.7347, 0.2653],
    [0.7347, 0.2653], [0.7347, 0.2653], [0.7347, 0.2653], [0.7347, 0.2653],
])

# Known gamut triangles in CIE xy (closed — last point repeats first)
_REC709 = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.64, 0.33]])
_REC2020 = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046], [0.708, 0.292]])
_P3D65 = np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060], [0.680, 0.320]])
_D65_WP = np.array([0.3127, 0.3290])


def rgb_to_cie_xy(rgb_pixels):
    """Convert RGB pixels (N×3, 0-1) to CIE xy chromaticity via XYZ."""
    xyz = rgb_pixels @ _RGB_TO_XYZ.T
    total = xyz[:, 0] + xyz[:, 1] + xyz[:, 2]
    valid = total > 1e-6
    x = np.zeros(len(rgb_pixels))
    y = np.zeros(len(rgb_pixels))
    x[valid] = xyz[valid, 0] / total[valid]
    y[valid] = xyz[valid, 1] / total[valid]
    return x[valid], y[valid]


def plot_chromaticity(pixels, title="Chromaticity Distribution", save_path=None):
    """Plot CIE xy chromaticity diagram with spectral locus and gamut triangles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cx, cy = rgb_to_cie_xy(pixels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Spectral locus
    locus_closed = np.vstack([_SPECTRAL_LOCUS, _SPECTRAL_LOCUS[0]])
    ax.fill(locus_closed[:, 0], locus_closed[:, 1], color='#1a1a2e', alpha=0.3)
    ax.plot(locus_closed[:, 0], locus_closed[:, 1], color='white', linewidth=1, alpha=0.5)

    # Gamut triangles
    ax.plot(_REC709[:, 0], _REC709[:, 1], 'c--', linewidth=1.5, label='Rec.709', alpha=0.8)
    ax.plot(_REC2020[:, 0], _REC2020[:, 1], 'g--', linewidth=1.5, label='Rec.2020', alpha=0.8)
    ax.plot(_P3D65[:, 0], _P3D65[:, 1], 'm--', linewidth=1.2, label='P3-D65', alpha=0.6)

    # D65 white point
    ax.plot(_D65_WP[0], _D65_WP[1], 'w+', markersize=12, markeredgewidth=2, label='D65')

    # Pixel chromaticity heatmap
    hist, xedges, yedges = np.histogram2d(cx, cy, bins=300, range=[[0, 0.8], [0, 0.9]])
    hist = np.log1p(hist * 10)
    ax.imshow(hist.T, origin='lower', extent=[0, 0.8, 0, 0.9], aspect='equal',
              cmap='hot', interpolation='nearest', alpha=0.9)

    # Re-draw triangles on top of heatmap
    ax.plot(_REC709[:, 0], _REC709[:, 1], 'c--', linewidth=1.5, alpha=0.8)
    ax.plot(_REC2020[:, 0], _REC2020[:, 1], 'g--', linewidth=1.5, alpha=0.8)
    ax.plot(_P3D65[:, 0], _P3D65[:, 1], 'm--', linewidth=1.2, alpha=0.6)
    ax.plot(_D65_WP[0], _D65_WP[1], 'w+', markersize=12, markeredgewidth=2)

    ax.set_xlabel('x', fontsize=12, color='white')
    ax.set_ylabel('y', fontsize=12, color='white')
    ax.set_title(title, fontsize=13, color='white', pad=12)
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9, loc='upper right', facecolor='black', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"  Plot saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gamut Wall Test — detect log vs display-referred footage')
    parser.add_argument('image', help='Path to image file (TIFF, PNG, EXR, etc.)')
    parser.add_argument('--plot', action='store_true', help='Generate chromaticity plot')
    parser.add_argument('--output', default='/tmp/gamut_wall', help='Output directory for plots')
    args = parser.parse_args()
    
    from processing import load_image
    
    print(f"Loading: {os.path.basename(args.image)}")
    frames = load_image(args.image)
    
    result = gamut_wall_test(frames)
    
    if args.plot:
        os.makedirs(args.output, exist_ok=True)
        px = frames[0].reshape(-1, 3)
        if len(px) > 500000:
            idx = np.random.RandomState(42).choice(len(px), 500000, replace=False)
            px = px[idx]
        
        name = os.path.splitext(os.path.basename(args.image))[0]
        verdict = "DISPLAY" if result['is_display_referred'] else "LOG"
        plot_chromaticity(px, 
                         title=f"{name} — {verdict} (conf: {result['confidence']:.2f})",
                         save_path=os.path.join(args.output, f"{name}_chromaticity.png"))
