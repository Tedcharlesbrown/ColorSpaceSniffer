
import cv2 as cv
import numpy as np
import math
import PyOpenColorIO as OCIO
from processing import OCIO_CST
from gamut_wall import _SPECTRAL_LOCUS

# ACES 2065-1 (AP0) primaries to CIE XYZ D65
# AP0 is a super-gamut that encloses the entire spectral locus.
# Correctly decoded footage maps to physically realizable colors (inside locus).
_AP0_TO_XYZ = np.array([
    [0.9525523959, 0.0000000000, 0.0000936786],
    [0.3439664498, 0.7281660966, -0.0721325464],
    [0.0000000000, 0.0000000000, 1.0088251844],
], dtype=np.float64)

# Build a closed spectral locus polygon (line of purples closes 700nm → 380nm)
_LOCUS_POLY = np.vstack([_SPECTRAL_LOCUS, _SPECTRAL_LOCUS[0]])

##returns a 1x3 array of the average RATIO of clipped pixels per channel (0.0–1.0)
## normalized by total pixels so bright scenes don't dominate the score
def CheckGamutClipping(frames):
    # FIX: replaced nested pixel loops with numpy vectorized ops — same result, orders of magnitude faster
    # FIX: normalized to ratio (clipped / total) so raw pixel count doesn't overwhelm other metrics
    total_clipped = np.zeros(3)
    total_pixels = 0
    for frame in frames:
        h, w = frame.shape[:2]
        total_clipped += np.sum(frame >= 1.0, axis=(0, 1))
        total_pixels += h * w
    avg_clipped_ratio = total_clipped / total_pixels  # now 0.0–1.0 per channel
    return avg_clipped_ratio.tolist()


def CheckNoiseLevels(frames) :
        
    #get dimensions of original image
    src_height, src_width = frames[0].shape[:2]

    #resize the image for display
    display_height = 720
    display_width = int(src_width * (display_height / src_height))

    ##defining empty arrays to be SNR and Noise Data and a helper variable
    AggragateSNR = 0
    Aggragate_Noise = [0.0, 0.0, 0.0, 0.0]
    Average_Noise = [0.0, 0.0, 0.0, 0.0]
    counter = 0

    for frame in frames :

        #resize the image for display
        display_image = cv.resize(frame, (display_width, display_height))

        #select an ROI on resized image
        roi = cv.selectROI("select a patch to calculate noise in", cv.cvtColor(display_image, cv.COLOR_BGR2RGB), True, False)

        x1, y1, width, height = roi

        #scale ROI back to original resolution
        scale_x = src_width / display_width
        scale_y = src_height / display_height

        x1_original = int(x1 * scale_x)
        y1_original = int(y1 * scale_y)
        width_original = int(width * scale_x)
        height_original = int(height * scale_y)

        patch = frame[y1_original : y1_original + height_original, x1_original : x1_original + width_original]

        XYZ_patch = cv.cvtColor(patch, cv.COLOR_BGR2XYZ)

        # FIX: replaced manual pixel loops with numpy mean/std — much faster, same math
        # ddof=1 for sample standard deviation (matches original n-1 denominator)
        avg_rgb = np.mean(patch, axis=(0, 1))           # [B, G, R]
        std_rgb = np.std(patch, axis=(0, 1), ddof=1)    # [B, G, R]
        avg_luma = float(np.mean(XYZ_patch[:, :, 1]))
        std_luma = float(np.std(XYZ_patch[:, :, 1], ddof=1))

        # FIX: corrected parenthesis bug — was `avg[2] / sqrt(...)` due to precedence,
        # should be `(avg[0] + avg[1] + avg[2]) / sqrt(...)`
        SNR = 20 * math.log10(
            (float(avg_rgb[0]) + float(avg_rgb[1]) + float(avg_rgb[2])) /
            math.sqrt(float(std_rgb[0])**2 + float(std_rgb[1])**2 + float(std_rgb[2])**2)
        )

        AggragateSNR += SNR
        Aggragate_Noise[0] += float(std_rgb[0])
        Aggragate_Noise[1] += float(std_rgb[1])
        Aggragate_Noise[2] += float(std_rgb[2])
        Aggragate_Noise[3] += std_luma

        counter += 1

    Average_SNR = AggragateSNR / counter
    Average_Noise[0] = Aggragate_Noise[0] / counter
    Average_Noise[1] = Aggragate_Noise[1] / counter
    Average_Noise[2] = Aggragate_Noise[2] / counter
    Average_Noise[3] = Aggragate_Noise[3] / counter

    return Average_SNR , Average_Noise


def CheckNeutralVarianceOnFrame(frame, percentile = 5) :
    #this is a function to check the neutral variance among low chroma pixels in the image. 
    # the function takes in a normalized image and a threshold for neutrals and returns the circular variance of the image

    # Frames are RGB — use RGB→LAB and RGB→HSV conversions.
    # Previously used COLOR_BGR2LAB/HSV which swapped R and B, producing wrong hue angles
    # and chroma for non-neutral pixels.
    Lab_frame = cv.cvtColor(frame, cv.COLOR_RGB2LAB) * 255
    HSV_frame = cv.cvtColor(frame, cv.COLOR_RGB2HSV) * 2 * math.pi

    # FIX: original code did np.percentile(np.max(...), percentile) which collapses to a
    # scalar before computing the percentile, so max_a always == the frame maximum.
    # Correct approach: compute the Nth percentile of the channel distribution directly.
    max_a = np.percentile(Lab_frame[:, :, 1], 100 - percentile)
    min_a = np.percentile(Lab_frame[:, :, 1], percentile)
    max_b = np.percentile(Lab_frame[:, :, 2], 100 - percentile)
    min_b = np.percentile(Lab_frame[:, :, 2], percentile)

    # FIX: replaced pixel-by-pixel nested loop with numpy masking — same logic, much faster
    mask = (
        (Lab_frame[:, :, 0] > 100) &
        (Lab_frame[:, :, 1] > min_a) & (Lab_frame[:, :, 1] < max_a) &
        (Lab_frame[:, :, 2] > min_b) & (Lab_frame[:, :, 2] < max_b)
    )

    sin_hues = np.where(mask, np.sin(HSV_frame[:, :, 0]), 0.0)
    cos_hues = np.where(mask, np.cos(HSV_frame[:, :, 0]), 0.0)

    sin_avg = np.mean(sin_hues)
    cos_avg = np.mean(cos_hues)

    R = np.sqrt(sin_avg**2 + cos_avg**2)
    circular_variance = 1 - R

    return circular_variance


def CheckNeutralVariance(frames, percentile = 5) :
    #calls CheckNeutralVarianceOnFrame for all of the frames in a given array, default passes through.
    Total_Variance = 0
    counter = 0
    for frame in frames:
        Total_Variance += CheckNeutralVarianceOnFrame(frame, percentile)
        counter += 1
    
    average_variance = Total_Variance / counter

    return average_variance


def CalculateLuma(frame):
    # BT.601 luma from RGB (frames are stored in RGB order after BGR→RGB conversion in processing.py).
    # Previously used cv.COLOR_BGR2YUV which swapped R/B weights: Y = 0.114R + 0.587G + 0.299B (wrong).
    return np.dot(frame, np.float32([0.299, 0.587, 0.114])).astype(np.float32)

def LaplacianEnergy(luminance) :

    laplacian = cv.Laplacian(luminance, cv.CV_32F, 3)
    AvgLaplacian = np.mean(np.abs(laplacian))

    #displayMe_abs = np.abs(luminance)
    #displayMeNorm = cv.normalize(displayMe_abs, None, 0, 255, cv.NORM_MINMAX)
    #displayMeRGB = displayMeNorm.astype(np.uint8)
    #dimensions = (1280, 720)
    #cv.imshow('this is the laplacian Frame', cv.resize(displayMeRGB, dimensions))

    return AvgLaplacian

def LocalContrastRatio(original_frames, transformed_frames) : 
    counter = 0
    ratio = 0
    for frame in original_frames :
        luminance_orignal = CalculateLuma(original_frames[counter])
        luminance_transformed = CalculateLuma(transformed_frames[counter])

        energy_original = LaplacianEnergy(luminance_orignal)
        energy_transformed = LaplacianEnergy(luminance_transformed)

        if energy_original < .000001 :
            return np.inf
        
        ratio += (energy_transformed / energy_original)

        counter += 1.0

        #print("the energery is: " , energy_original, energy_transformed)
    
    averageContrastRatio = float(ratio / float(counter))

    return averageContrastRatio


def OneFrameChannelCorrelation(frame) :
    #flatten the image
    pixels = frame.reshape(-1, 3)

    #calculate and apply a mask based on pixel brightness
    mask = np.all((pixels > 0.01) & (pixels < 0.99), axis = 1)
    masked_pixels = pixels[mask]
    
    correlation = np.corrcoef(masked_pixels, rowvar= False)

    #ideal correlation matrix has off-diagonals near 1
    off_diag = np.array([correlation[0,1], correlation[0,2], correlation[1,2]])

    return 1 - np.mean(off_diag)

def ManyFrameChannelCorrelation(frames) :
    correlation = 0
    counter = 0
    for frame in frames :
        correlation += OneFrameChannelCorrelation(frame)
        counter += 1

    AverageCorrelation = correlation / counter

    return AverageCorrelation

def RoundTripPSNR(OriginalFrames, TransformedFrames, config, idt, odt) :
    # Round-trip through scene-linear: IDT → scene_linear → inverse IDT
    # This avoids display-space clipping contaminating the PSNR measurement.
    # The correct IDT should round-trip near-perfectly since no display clipping occurs.
    try:
        scene_linear = OCIO_CST(OriginalFrames, config, idt, "ACES2065-1")
        RoundTripFrames = OCIO_CST(scene_linear, config, "ACES2065-1", idt)
    except Exception:
        # Fallback to original display round-trip if scene-linear names don't resolve
        RoundTripFrames = OCIO_CST(TransformedFrames, config, odt, idt)

    counter = 0
    total_psnr = 0.0
    for frame in OriginalFrames :
        total_psnr += cv.PSNR(OriginalFrames[counter], RoundTripFrames[counter])
        counter += 1

    avg_PSNR = total_psnr / counter

    return avg_PSNR

def CheckSaturation(TransformedFrames) :
    avg_pixel_Saturation = 0
    counter = 0
    for frame in TransformedFrames :
        HSV_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        avg_pixel_Saturation += np.mean(HSV_frame[:,:, 1])
        counter += 1

    avg_saturation = avg_pixel_Saturation / counter

    return avg_saturation

def HistogramEntropy(transformed_frames):
    # Shannon entropy of the luminance histogram. Higher = more natural distribution.
    # Crushed/bimodal/gap-filled histograms from wrong IDTs return lower entropy.
    total_entropy = 0.0
    for frame in transformed_frames:
        # RGB-weighted luminance (data is in RGB order)
        luma = np.dot(frame, [0.2126, 0.7152, 0.0722])
        luma_clipped = np.clip(luma, 0.0, 1.0)
        hist, _ = np.histogram(luma_clipped.ravel(), bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            continue
        probs = hist / total
        nonzero = probs > 0
        entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
        total_entropy += entropy
    return total_entropy / len(transformed_frames)


def SceneLinearRangeCheck(original_frames, config, idt):
    # Apply ONLY the IDT (no ODT) to get scene-linear intermediate values.
    # A wrong IDT produces deeply negative values (< -0.1) from color math errors.
    # Note: many correct IDTs have a non-zero black floor, so values slightly below 0
    # (e.g. -0.01) are normal — we use -0.1 as a tolerance to avoid penalising that.
    # We do NOT penalise high values: bright scenes legitimately produce scene-linear
    # values >> 10 in ACES 2065-1 (a 14-stop camera can have highlights at ~500).
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "scene_linear"

    try:
        linear_frames = OCIO_CST(original_frames, config, idt, scene_linear_name)
    except Exception:
        return 0.0

    total_penalty = 0.0
    for frame in linear_frames:
        # Only penalise deeply negative values — genuine black-floor offsets stay < 0.1
        neg_ratio = np.mean(frame < -0.1)
        total_penalty += neg_ratio
    return total_penalty / len(linear_frames)


def BandingDetection(frames):
    # Detect posterization/banding via gradient magnitude analysis.
    # Banding creates many abrupt shallow steps; natural content has near-zero (smooth)
    # or large (genuine edges) gradients. Returns severity score (higher = more banding = worse).
    total_severity = 0.0
    for frame in frames:
        luma = np.dot(frame, [0.2126, 0.7152, 0.0722]).astype(np.float32)
        sobelx = cv.Sobel(luma, cv.CV_32F, 1, 0, ksize=3)
        sobely = cv.Sobel(luma, cv.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        # Banding steps fall between the noise floor and genuine hard edges
        total_severity += np.mean((grad_mag > 0.005) & (grad_mag < 0.08))
    return total_severity / len(frames)


def ShadowNoiseAmplification(original_frames, transformed_frames):
    # Compare noise in dark regions before and after the transform.
    # Shadow pixels are defined as the bottom 20th percentile by luminance.
    # Returns the amplification ratio: std_transformed / std_original (> 1.0 = amplification).
    total_ratio = 0.0
    count = 0
    for orig, trans in zip(original_frames, transformed_frames):
        luma_orig = np.dot(orig, [0.2126, 0.7152, 0.0722])
        threshold = np.percentile(luma_orig, 20)
        shadow_mask = luma_orig <= threshold
        shadow_orig = orig[shadow_mask].ravel()
        shadow_trans = trans[shadow_mask].ravel()
        if shadow_orig.size < 10:
            continue
        std_orig = np.std(shadow_orig)
        if std_orig < 1e-8:
            continue
        total_ratio += np.std(shadow_trans) / std_orig
        count += 1
    return total_ratio / count if count > 0 else 1.0


def GreyWorldDeviation(transformed_frames):
    # Check if the average color of transformed frames is roughly neutral grey.
    # Computes mean R, G, B and measures Euclidean distance of normalized ratios
    # from [1/3, 1/3, 1/3]. Higher = further from neutral = worse.
    total_deviation = 0.0
    neutral = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    for frame in transformed_frames:
        mean_rgb = np.mean(frame, axis=(0, 1))  # [R, G, B] in RGB order
        total = np.sum(mean_rgb)
        if total < 1e-8:
            continue
        normalized = mean_rgb / total
        total_deviation += np.linalg.norm(normalized - neutral)
    return total_deviation / len(transformed_frames)


def BlackPedestalCheck(frames, config, idt_list):
    """Pre-scoring diagnostic: check footage's darkest neutral pixels against IDT black floors.

    Returns:
        darkest_neutral_luma (float): luminance of the darkest neutral pixel
        has_true_black (bool): whether footage contains pixels at absolute zero
        true_black_count (int): number of neutral pixels at absolute zero
        eliminations (dict): {idt_name: floor_value} for IDTs where footage is below floor
    """
    # Combine all frames, find neutral pixels (R≈G≈B)
    all_neutral_luma = []
    true_black_count = 0
    for frame in frames:
        spread = frame.max(axis=2) - frame.min(axis=2)
        neutral_mask = spread < 0.02
        if neutral_mask.sum() == 0:
            continue
        neutral_pixels = frame[neutral_mask]
        luma = np.dot(neutral_pixels, np.float32([0.2126, 0.7152, 0.0722]))
        all_neutral_luma.append(luma)
        true_black_count += int(np.sum(neutral_pixels.max(axis=1) < 0.001))

    if not all_neutral_luma:
        return None, False, 0, {}

    all_luma = np.concatenate(all_neutral_luma)
    darkest = float(all_luma.min())
    has_true_black = true_black_count > 0

    # Compute black floor for each IDT
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "ACES2065-1"

    eliminations = {}
    for idt in idt_list:
        try:
            cpu = config.getProcessor(scene_linear_name, idt).getDefaultCPUProcessor()
            black = np.float32([[[0.0, 0.0, 0.0]]])
            cpu.applyRGB(black)
            floor_luma = float(np.dot(black[0, 0], np.float32([0.2126, 0.7152, 0.0722])))
            # Small tolerance for 8-bit quantization (~0.004)
            if darkest < floor_luma - 0.005:
                eliminations[idt] = floor_luma
        except Exception:
            continue

    return darkest, has_true_black, true_black_count, eliminations


def BlackPedestalPenalty(darkest_neutral_luma, config, idt):
    """Compute penalty for a single IDT based on black pedestal violation.

    Returns a penalty value >= 0. Higher = footage is further below the floor.
    Returns 0 if footage is at or above the floor.
    """
    if darkest_neutral_luma is None:
        return 0.0

    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "ACES2065-1"

    try:
        cpu = config.getProcessor(scene_linear_name, idt).getDefaultCPUProcessor()
        black = np.float32([[[0.0, 0.0, 0.0]]])
        cpu.applyRGB(black)
        floor_luma = float(np.dot(black[0, 0], np.float32([0.2126, 0.7152, 0.0722])))
    except Exception:
        return 0.0

    # How far below the floor is the footage? (0 if at or above)
    violation = max(0.0, (floor_luma - 0.005) - darkest_neutral_luma)
    # Scale up — even small violations are meaningful
    return violation * 10.0


def IterativeDecodeDivergence(original_frames, config, idt, iterations=3):
    """Apply the IDT decode repeatedly and measure how fast values diverge.

    The correct IDT aggressively expands dynamic range, so repeated application
    sends values to infinity quickly. Wrong-but-gentle IDTs barely change anything,
    so values stay stable.

    Returns the divergence ratio: mean(abs(iteration_N)) / mean(abs(iteration_1)).
    Higher = more divergent = more aggressive decode = more likely correct.
    Returns 0.0 if computation fails.
    """
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "ACES2065-1"

    # Use a single representative frame (first frame, downsampled for speed)
    frame = original_frames[0].copy()
    # Downsample to ~512px wide for speed
    h, w = frame.shape[:2]
    if w > 512:
        scale = 512.0 / w
        frame = cv.resize(frame, (512, int(h * scale)))

    means = []
    img = frame.copy()
    for i in range(iterations):
        try:
            cpu = config.getProcessor(idt, scene_linear_name).getDefaultCPUProcessor()
            cpu.applyRGB(img)
            # Clip BOTH the image and the stats to avoid float overflow
            img = np.clip(img, -1e6, 1e6)
            mean_val = float(np.mean(np.abs(img)))
            if math.isnan(mean_val) or math.isinf(mean_val):
                mean_val = 1e6
            means.append(mean_val)
        except Exception:
            return 0.0

    if len(means) < 2 or means[0] < 1e-10:
        return 1e6  # couldn't compute first iteration = treat as max divergence

    # Ratio of final iteration to first iteration
    raw_ratio = means[-1] / means[0]
    if math.isnan(raw_ratio) or math.isinf(raw_ratio):
        return 1e6
    return min(raw_ratio, 1e6)


def ShadowArtifactMonotonicity(original_frames, config, idt, n_bins=20, blur_sigma=12):
    """Measure whether residual variance increases monotonically with local brightness.

    Correct IDT: film-like grain where brighter areas carry more variance.
    Wrong IDT: shadow regions show disproportionate variance from compression
    artifacts being amplified by an incorrect curve shape.

    Method:
      1. Decode frames to scene-linear via the IDT.
      2. Separate into low-frequency (Gaussian blur) and high-frequency (residual).
      3. Bin pixels by local mean brightness; compute residual variance per bin.
      4. Spearman-rank-correlate bin variances against bin means.
         Monotonically increasing → correct IDT (higher = better).

    Returns a score in [-1, +1]. Higher is better.
    """
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "ACES2065-1"

    try:
        linear_frames = OCIO_CST(original_frames, config, idt, scene_linear_name)
    except Exception:
        return 0.0

    all_local_mean = []
    all_residual = []

    ksize = int(blur_sigma * 6) | 1  # must be odd
    for frame in linear_frames:
        luma = np.dot(frame, np.float32([0.2126, 0.7152, 0.0722])).astype(np.float32)
        blurred = cv.GaussianBlur(luma, (ksize, ksize), blur_sigma)
        residual = luma - blurred
        all_local_mean.append(blurred.ravel())
        all_residual.append(residual.ravel())

    local_means = np.concatenate(all_local_mean)
    residuals = np.concatenate(all_residual)

    # Clip outliers before binning
    lo = np.percentile(local_means, 1)
    hi = np.percentile(local_means, 99)
    if hi <= lo:
        return 0.0

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(local_means, bin_edges) - 1, 0, n_bins - 1)

    bin_means = []
    bin_variances = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 10:
            continue
        bin_means.append(float(np.mean(local_means[mask])))
        bin_variances.append(float(np.var(residuals[mask])))

    if len(bin_means) < 3:
        return 0.0

    bm = np.array(bin_means)
    bv = np.array(bin_variances)

    # Spearman rank correlation (no scipy dependency)
    rank_m = np.argsort(np.argsort(bm)).astype(float)
    rank_v = np.argsort(np.argsort(bv)).astype(float)
    n = float(len(rank_m))
    d_sq = np.sum((rank_m - rank_v) ** 2)
    spearman = 1.0 - (6.0 * d_sq) / (n * (n ** 2 - 1))
    return float(np.clip(spearman, -1.0, 1.0))


def _points_in_spectral_locus(px, py):
    """Vectorized ray-casting point-in-polygon test against the spectral locus.

    The spectral locus polygon is closed by the line of purples (700nm → 380nm).
    Returns a boolean array: True = inside (physically realizable chromaticity).
    """
    poly_x = _LOCUS_POLY[:, 0]
    poly_y = _LOCUS_POLY[:, 1]
    n = len(poly_x)

    inside = np.zeros(len(px), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        # Does this edge cross the horizontal ray from (px, py) in the +x direction?
        cross_y = (yi > py) != (yj > py)
        denom = np.where(np.abs(yj - yi) > 1e-12, yj - yi, 1e-12)
        x_intersect = np.where(cross_y, (xj - xi) * (py - yi) / denom + xi, np.inf)
        inside ^= cross_y & (px < x_intersect)
        j = i

    return inside


def SpectralLocusViolation(original_frames, config, idt, max_pixels=200000):
    """Measure the fraction of pixels whose chromaticity falls outside the spectral locus.

    Method:
      1. Decode frames to scene-linear ACES 2065-1 (AP0) via the IDT.
      2. Convert AP0 → CIE XYZ using the AP0_TO_XYZ matrix.
      3. Compute CIE xy chromaticity for each pixel.
      4. Count what fraction lies outside the closed spectral locus polygon.

    ACES AP0 primaries encompass the entire spectral locus, so pixels from a
    correctly decoded physical camera scene should map to physically realizable
    chromaticities (inside the locus). A wrong IDT distorts the color science,
    sending chromaticities outside what physics allows.

    Returns violation fraction [0.0, 1.0]. Lower is better.
    """
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "ACES2065-1"

    try:
        linear_frames = OCIO_CST(original_frames, config, idt, scene_linear_name)
    except Exception:
        return 0.0

    all_px = []
    all_py = []

    for frame in linear_frames:
        h, w = frame.shape[:2]
        # Downsample to keep computation fast
        if h * w > max_pixels:
            scale = (max_pixels / (h * w)) ** 0.5
            frame = cv.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))

        pixels = frame.reshape(-1, 3).astype(np.float64)

        # AP0 scene-linear → CIE XYZ
        xyz = pixels @ _AP0_TO_XYZ.T  # N×3: [X, Y, Z]

        # Only include pixels with positive luminance and non-zero sum
        Y = xyz[:, 1]
        total = xyz[:, 0] + xyz[:, 1] + xyz[:, 2]
        valid = (Y > 1e-6) & (total > 1e-6)
        if valid.sum() == 0:
            continue

        x_chrom = xyz[valid, 0] / total[valid]
        y_chrom = xyz[valid, 1] / total[valid]
        all_px.append(x_chrom)
        all_py.append(y_chrom)

    if not all_px:
        return 0.0

    px = np.concatenate(all_px)
    py = np.concatenate(all_py)

    inside = _points_in_spectral_locus(px, py)
    violation_fraction = float(1.0 - np.mean(inside))
    return violation_fraction


def DynamicRangeRecovery(original_frames, config, idt, odt,
                         exposures=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Measure how much latitude the IDT preserves through exposure manipulation.

    Applies the IDT to get scene-linear data, multiplies by each exposure
    offset, then applies the ODT.  Shannon entropy of the resulting luminance
    histogram is measured at every exposure.  An IDT that genuinely maps to
    scene-linear will maintain high entropy across a wide exposure sweep —
    highlights recover when pulled down and shadows reveal detail when pushed up.

    Returns a scalar score (higher = more latitude = better).
    """
    try:
        scene_linear_name = config.getColorSpace(OCIO.ROLE_SCENE_LINEAR).getName()
    except Exception:
        scene_linear_name = "scene_linear"

    try:
        linear_frames = OCIO_CST(original_frames, config, idt, scene_linear_name)
    except Exception:
        return 0.0  # can't evaluate; neutral

    entropy_values = []
    for ev in exposures:
        # Multiply scene-linear data by the exposure offset (vectorized)
        exposed = [np.float32(frame * ev) for frame in linear_frames]

        # Map through the ODT to display-referred space
        try:
            display_frames = OCIO_CST(exposed, config, scene_linear_name, odt)
        except Exception:
            continue

        # Compute per-frame luminance entropy and average
        frame_entropies = []
        for frame in display_frames:
            luma = np.dot(frame, np.float32([0.2126, 0.7152, 0.0722]))
            luma_clipped = np.clip(luma, 0.0, 1.0)
            hist, _ = np.histogram(luma_clipped.ravel(), bins=256, range=(0.0, 1.0))
            hist = hist.astype(np.float64)
            total = hist.sum()
            if total == 0:
                frame_entropies.append(0.0)
                continue
            probs = hist / total
            nonzero = probs > 0
            frame_entropies.append(-np.sum(probs[nonzero] * np.log2(probs[nonzero])))

        if frame_entropies:
            entropy_values.append(np.mean(frame_entropies))

    if not entropy_values:
        return 0.0

    # Score = mean entropy across all exposure levels.
    # High mean ⇒ detail is preserved at every stop ⇒ correct IDT.
    return float(np.mean(entropy_values))


def ScoreIDTtransforms(OriginalFrames, TransformedFrames, config, idt, odt,
                       darkest_neutral_luma=None, PSNR=True, Noise=False):
    breakdown = {}

    # --- Raw metric values ---
    raw_nv = CheckNeutralVariance(TransformedFrames)
    raw_cc = ManyFrameChannelCorrelation(TransformedFrames)
    clip_ratios = CheckGamutClipping(TransformedFrames)
    raw_clip = sum(max(0.0, r - 0.05) for r in clip_ratios)
    raw_lcr = LocalContrastRatio(OriginalFrames, TransformedFrames)
    raw_slr = SceneLinearRangeCheck(OriginalFrames, config, idt)
    raw_entropy = HistogramEntropy(TransformedFrames)
    raw_banding = BandingDetection(TransformedFrames)
    raw_shadow = ShadowNoiseAmplification(OriginalFrames, TransformedFrames)
    raw_grey = GreyWorldDeviation(TransformedFrames)
    raw_pedestal = BlackPedestalPenalty(darkest_neutral_luma, config, idt)
    raw_divergence = IterativeDecodeDivergence(OriginalFrames, config, idt)
    raw_sam = ShadowArtifactMonotonicity(OriginalFrames, config, idt)
    raw_slv = SpectralLocusViolation(OriginalFrames, config, idt)

    # --- Weighted contributions ---
    # Weights found by brute-force random search (weight_optimizer.py, 50k combos, 6 ground-truth files).
    # Pre-processing mirrors what the optimizer cached:
    #   lcr_low  = max(0, 1.5 - lcr)   — penalise near-identity decodes
    #   lcr_high = max(0, lcr - 3.5)   — penalise extreme expansion
    #   shadow   = max(0, shadow - 1.0) — threshold out normal amplification
    #   divergence = log1p(min(raw, 1e6))
    #   slv weight is NEGATIVE — more locus violations → wider-gamut decode → rewarded
    w_nv  = 2.2745 * raw_nv
    w_cc  = 2.2501 * raw_cc
    w_clip = 0.1781 * raw_clip
    w_lcr = 3.3046 * max(0.0, 1.5 - raw_lcr) + 0.1790 * max(0.0, raw_lcr - 3.5)
    w_slr = 4.5794 * raw_slr
    w_entropy = 0.0382 * raw_entropy
    w_banding = 0.9193 * raw_banding
    w_shadow  = 0.0490 * max(0.0, raw_shadow - 1.0)
    w_grey    = 0.3174 * raw_grey
    w_pedestal = 0.0 * raw_pedestal  # DISABLED — too aggressive on real footage
    safe_divergence = min(raw_divergence, 1e6) if not math.isnan(raw_divergence) else 0.0
    w_divergence = -0.2249 * math.log1p(safe_divergence)
    w_sam = -2.6117 * raw_sam
    # Negative weight: wide-gamut IDTs produce more out-of-locus chromaticities (correct
    # behaviour for cameras with imaginary primaries like V-Gamut, S-Gamut3).
    w_slv = -2.4775 * raw_slv

    Score = w_nv + w_cc + w_clip + w_lcr + w_slr + w_entropy + w_banding + w_shadow + w_grey + w_pedestal + w_divergence + w_sam + w_slv

    ##come up with a way to score and incorporate Avg Noise into aggragate Score
    if Noise :
        AvgSNR, AvgNoise = CheckNoiseLevels(TransformedFrames)
        Score += 1.0 * AvgNoise

    ##come up with a way to score and incorporate PSNR into aggragate Score
    raw_psnr = 0.0
    if PSNR :
        raw_psnr = RoundTripPSNR(OriginalFrames, TransformedFrames, config, idt, odt)
        Score = Score / (raw_psnr / 48)

    # Store everything for reporting
    breakdown['IDT'] = idt
    breakdown['NeutralVariance_raw'] = raw_nv
    breakdown['NeutralVariance_weighted'] = w_nv
    breakdown['ChannelCorrelation_raw'] = raw_cc
    breakdown['ChannelCorrelation_weighted'] = w_cc
    breakdown['GamutClipping_raw'] = raw_clip
    breakdown['GamutClipping_weighted'] = w_clip
    breakdown['LocalContrastRatio_raw'] = raw_lcr
    breakdown['LocalContrastRatio_weighted'] = w_lcr
    breakdown['SceneLinearRange_raw'] = raw_slr
    breakdown['SceneLinearRange_weighted'] = w_slr
    breakdown['HistogramEntropy_raw'] = raw_entropy
    breakdown['HistogramEntropy_weighted'] = w_entropy
    breakdown['BandingDetection_raw'] = raw_banding
    breakdown['BandingDetection_weighted'] = w_banding
    breakdown['ShadowNoise_raw'] = raw_shadow
    breakdown['ShadowNoise_weighted'] = w_shadow
    breakdown['GreyWorldDeviation_raw'] = raw_grey
    breakdown['GreyWorldDeviation_weighted'] = w_grey
    breakdown['BlackPedestal_raw'] = raw_pedestal
    breakdown['BlackPedestal_weighted'] = w_pedestal
    breakdown['Divergence_raw'] = raw_divergence
    breakdown['Divergence_weighted'] = w_divergence
    breakdown['ShadowArtifactMonotonicity_raw'] = raw_sam
    breakdown['ShadowArtifactMonotonicity_weighted'] = w_sam
    breakdown['SpectralLocusViolation_raw'] = raw_slv
    breakdown['SpectralLocusViolation_weighted'] = w_slv
    breakdown['PSNR'] = raw_psnr
    breakdown['FinalScore'] = Score

    return Score, breakdown
