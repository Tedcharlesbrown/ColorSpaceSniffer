
import cv2 as cv
import numpy as np
import math
from processing import OCIO_CST

##returns a 1x3 array of the average number of clipped pixels per channel (BGR)
def CheckGamutClipping(frames):
    # FIX: replaced nested pixel loops with numpy vectorized ops — same result, orders of magnitude faster
    total_clipped = np.zeros(3)
    for frame in frames:
        total_clipped += np.sum(frame >= 1.0, axis=(0, 1))
    avg_clipped = total_clipped / len(frames)
    return avg_clipped.tolist()


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

    #convert to LAB to do thresholding, convert to HSV to find hue angles 
    Lab_frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB) * 255
    HSV_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) * 2 * math.pi

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


def CalculateLuma(frame) :

    YUVFrame = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
    Luma = YUVFrame[:,:, 0]

    return Luma

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

def ScoreIDTtransforms(OriginalFrames, TransformedFrames, config, idt, odt, PSNR = True, Noise = False):
    Score = 0.0

    Score += 2.0 * CheckNeutralVariance(TransformedFrames)

    Score += 1.5 * ManyFrameChannelCorrelation(TransformedFrames)

    Score += 1.0 * sum(CheckGamutClipping(TransformedFrames))

    Score += 1.0 * LocalContrastRatio(OriginalFrames, TransformedFrames)
    
    ##come up with a way to score and incorporate Avg Noise into aggragate Score
    if Noise :
        AvgSNR, AvgNoise = CheckNoiseLevels(TransformedFrames)
        Score += 1.0 * AvgNoise

    ##come up with a way to score and incorporate PSNR into aggragate Score
    if PSNR :
        psnrScore = RoundTripPSNR(OriginalFrames, TransformedFrames, config, idt, odt)
        Score = Score / (psnrScore / 48)

    Average_saturation = CheckSaturation(TransformedFrames)
    if Average_saturation * 10 < 1:
        Score += Score * Average_saturation * 10
    else :
        Score += 0

    return Score
