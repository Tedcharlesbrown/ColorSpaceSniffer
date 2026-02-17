

import cv2 as cv
import numpy as np
import subprocess
import PyOpenColorIO as OCIO
from tkinter import filedialog, Tk  
import math
import os


def encode_hap_video(input_file, output_file):
    subprocess.run([
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'hap',
        output_file
    ])

##function to extract a number of frames from a video based on sample count, default is 5
## returns an array of frames, normalized between 0 and 1
def extract_frames(VideoPath, SampleCount = 5):
    VideoFile = cv.VideoCapture(VideoPath)
    FrameCount = VideoFile.get(cv.CAP_PROP_FRAME_COUNT)
    Interval = FrameCount / SampleCount
    Frames = []

    if SampleCount == 1 :
        VideoFile.set(cv.CAP_PROP_POS_FRAMES, FrameCount / 2)
        ret , rawFrame = VideoFile.read()

        frame_rgb = cv.cvtColor(rawFrame, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        #RGB_frame = cv.cvtColor(Frame, cv.COLOR_BGR2RGB)

        Frames.append(frame_rgb)

    else :
        for sample in  range(SampleCount) :
            VideoFile.set(cv.CAP_PROP_POS_FRAMES, Interval * sample - 1)
            ret , rawFrame = VideoFile.read()

            frame_rgb = cv.cvtColor(rawFrame, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

            #RGB_frame = cv.cvtColor(Frame, cv.COLOR_BGR2RGB)

            Frames.append(frame_rgb)
        
    VideoFile.release()

    return Frames



##a better function to extract a number of frames from a video at 16bit depth based on sample count, default is 5
#extraction returns an array of 16 bit images with N entries in it, images are BGR
def better_frame_extraction(VideoPath, SampleCount = 5):
    VideoFile = cv.VideoCapture(VideoPath)
    FrameCount = VideoFile.get(cv.CAP_PROP_FRAME_COUNT)
    Interval = FrameCount / SampleCount
    Frames = []

    #calling FFMPEG to extract N number of frames from the video at the desired interval at 16 bit depth
    for frame in range(SampleCount):
        ThisFrame = int(Interval * frame)
        ThisFrameFlag = f"select=eq(n\\,{ThisFrame})"
        outputFileName = 'IntermediateFrame' + str(frame) + '.tiff'
        subprocess.run([
            'ffmpeg', '-y',
            '-i', VideoPath, '-map', '0:v:0', '-vf', ThisFrameFlag, '-vframes', '1','-c:v', 'tiff' , '-pix_fmt', 'rgb48le', outputFileName
        ])
            #flag used to be 'rgb48le' swapped for testing
    #reading in all of the intermediate frames, converting them to RGB, and then normalizing them
    for frame in range(SampleCount):
        inputFileName = 'IntermediateFrame' + str(frame) + '.tiff'        
        uint16Frame = (cv.imread(inputFileName, cv.IMREAD_UNCHANGED))
        #RGB_Frame = cv.cvtColor(uint16Frame, cv.COLOR_YUV2BGR)
        float_frame = uint16Frame.astype(np.float32) / (pow(2, 16) - 1)
        Frames.append(float_frame)

    VideoFile.release()
    return Frames


def OCIO_CST(Frames, config, IDT, ODT):
    Transformed_Frames = []
    Processor = config.getProcessor(IDT, ODT)
    CPU_Processor = Processor.getDefaultCPUProcessor()
    #print(Frames[0].dtype)
    for frame in Frames :
        frame_copy = frame.copy()
        if frame_copy.shape[2] == 3:
            CPU_Processor.applyRGB(frame_copy)
            #print("Did apply RGB")
        if frame_copy.shape[2] == 4 :
            CPU_Processor.applyRGBA(frame_copy)
            print('Did apply RGBA')
        Transformed_Frames.append(frame_copy)
    return Transformed_Frames

##returns a 1x3 array of the average number of clipped pixels per channel (BGR)
def CheckGamutClipping(frames):
    rows , cols = frames[0].shape[:2]
    NumClippedPixels = [0, 0, 0]
    AvgClippedPixPerFrame = [0, 0, 0]
    counter = 0
    for frame in frames :
        for r in range(rows):
            for c in range(cols):
                    NumClippedPixels[0] += int(frames[counter][r,c,0])
                    NumClippedPixels[1] += int(frames[counter][r,c,1])
                    NumClippedPixels[2] += int(frames[counter][r,c,2])
        counter += 1

    AvgClippedPixPerFrame[0] = NumClippedPixels[0] / counter
    AvgClippedPixPerFrame[1] = NumClippedPixels[1] / counter
    AvgClippedPixPerFrame[2] = NumClippedPixels[2] / counter
    return AvgClippedPixPerFrame


def CheckNoiseLevels(frames) :
        
    #get dimensions of original image
    src_height, src_width = frames[0].shape[:2]

    #resizr the image for display
    display_height = 720
    display_width = int(src_width * (display_height / src_height))


    ##defining empty arrays to be SNR and Noise Data and a helper variable
    AggragateSNR = 0
    Aggragate_Noise = [0, 0, 0, 0]
    Average_Noise = [0,0,0,0]
    counter = 0

    for frame in frames :

        #resizr the image for display
        display_image = cv.resize(frame, (display_width, display_height))

        #select an ROI on resized image
        roi = cv.selectROI("select a patch to calculate noise in", cv.cvtColor(display_image, cv.COLOR_BGR2RGB), True, False)

        x1, y1, width, height = roi

        #scale ROI back to orignial resolution
        scale_x = src_width / display_width
        scale_y = src_height / display_height

        x1_original = int(x1 * scale_x)
        y1_original = int(y1 * scale_y)
        width_original = int(width * scale_x)
        height_original = int(height * scale_y)

        patch = frame[y1_original : y1_original + height_original, x1_original : x1_original + width_original]



        XYZ_patch = cv.cvtColor(patch, cv.COLOR_BGR2XYZ)

        rows, cols, chan = patch.shape
        n = cols * rows

        sum = [0, 0, 0, 0]

        #calculating average
        for x in range(rows):
            for y in range(cols):
                sum[0] += patch[x, y, 0]
                sum[1] += patch[x, y, 1]
                sum[2] += patch[x, y, 2]
                sum[3] += XYZ_patch[x,y,1]

        avg = [sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n]

        #calculating numerator for standard deviation (noise)
        std_numerator = [0,0,0,0]
        for x in range(rows):
            for y in range(cols):
                std_numerator[0] += (patch[x, y, 0] - avg[0])**2
                std_numerator[1] += (patch[x, y, 1] - avg[1])**2
                std_numerator[2] += (patch[x, y, 2] - avg[2])**2
                std_numerator[3] += (XYZ_patch[x, y, 1] - avg[3])**2

        std_dev = [0,0,0,0]

        std_dev[0] = math.sqrt(std_numerator[0] / (n - 1))
        std_dev[1] = math.sqrt(std_numerator[1] / (n - 1))
        std_dev[2] = math.sqrt(std_numerator[2] / (n - 1))
        std_dev[3] = math.sqrt(std_numerator[3] / (n - 1))

        SNR = 20 * math.log10((avg[0] + avg[1] + avg[2] / (math.sqrt(std_dev[0]**2 + std_dev[1]**2 + std_dev[2]**2))))

        AggragateSNR += SNR
        Aggragate_Noise[0] += std_dev[0]
        Aggragate_Noise[1] += std_dev[1]
        Aggragate_Noise[2] += std_dev[2]
        Aggragate_Noise[3] += std_dev[3]


        counter += 1

    Average_SNR = AggragateSNR / counter
    Average_Noise[0] = Aggragate_Noise[0] / counter
    Average_Noise[1] = Aggragate_Noise[1] / counter
    Average_Noise[2] = Aggragate_Noise[2] / counter
    Average_Noise[3] = Aggragate_Noise[3] / counter

    return Average_SNR , Average_Noise


def CheckNeutralVarianceOnFrame(frame, percentile = 5) :
    #this is a function to check the neutral variance among low chroma pixels in the image. 
    # the function takes in a normalized imgae and a threshold for neutrals and returns the circular variance of the image

    #convert to LAB to do thresholding, convert to HSV to find hue angles 
    Lab_frame = cv.cvtColor(frame, cv.COLOR_BGR2LAB) * 255
    HSV_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV) * 2 * math.pi

    #defining htreshold percentages
    max_a = np.percentile(np.max(Lab_frame[:,:,1]), percentile)
    min_a = np.percentile(np.min(Lab_frame[:,:,1]), percentile)
    max_b = np.percentile(np.max(Lab_frame[:,:,2]), percentile)
    min_b = np.percentile(np.min(Lab_frame[:,:,2]), percentile)
    row , col = frame.shape[:2]
    sin_hues = np.zeros((row, col, 1), np.float32)
    cos_hues = np.zeros((row, col, 1), np.float32)
    for r in range(row) :
        for c in range(col) :
            if  Lab_frame[r, c, 0] > 100 and min_a < Lab_frame[r, c, 1] < max_a and min_b < Lab_frame[r, c, 2] < max_b :
                sin_hues[r, c] = np.sin(HSV_frame[r,c, 0])
                cos_hues[r, c] = np.cos(HSV_frame[r,c, 0])
 
    sin_avg = np.mean(sin_hues)
    cos_avg = np.mean(cos_hues)

    R = np.sqrt(sin_avg**2 + cos_avg**2)
    circular_variance = 1 - R

    return circular_variance


def CheckNeutralVariance(frames, percentile = 5) :
    #calls check beutral variance on frame for all of the frames in a given array, default passes through.
    Total_Variance = 0
    counter = 0
    for frame in frames:
        Total_Variance += CheckNeutralVarianceOnFrame(frame, percentile)
        counter += 1
    
    average_variance = Total_Variance / counter

    return average_variance


def CalculateLuma(frame) :

    #luminance = 0.2126 * frame[:,:, 2] + 0.7152 * frame[:,:,1] + 0.0722 * frame[:,:,0]
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

    #clcualte and apply a mask based on pixel brightness
    mask = np.all((pixels > 0.01) & (pixels < 0.99), axis = 1)
    masked_pixels = pixels[mask]

    #if len(masked_pixels) < 1000:
    #    return np.inf
    
    correlation = np.corrcoef(masked_pixels, rowvar= False)

    #ideal correlation matrix has off-diagnols near 1
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


def main():
    ###Setting the OCIO Config 
    config = OCIO.Config.CreateFromFile(r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\studio-config-all-views-v4.0.0_aces-v2.0_ocio-v2.5.ocio")


    ###Setting the OCIO transforms to be appraised during this appraisal    
    IDTs = []

    DJI_IDT = "D-Log D-Gamut"
    IDTs.append(DJI_IDT)
    Apple_IDT = "Apple Log"
    IDTs.append(Apple_IDT)
    LogC3_IDT = "ARRI LogC3 (EI800)"
    IDTs.append(LogC3_IDT)
    #AWG3_IDT = "Linear ARRI Wide Gamut 3"
    #IDTs.append(AWG3_IDT)
    LogC4_IDT = "ARRI LogC4"
    IDTs.append(LogC4_IDT)
    #AWG4_IDT = "Linear ARRI Wide Gamut 4"
    #IDTs.append(AWG4_IDT)
#    BMDLog_IDT = "BMDFilm WideGamut Gen5"
#    IDTs.append(BMDLog_IDT)
    #BMDLin_IDT = "Linear BMD WideGamut Gen5"
    #IDTs.append(BMDLin_IDT)
    CLog2_IDT = "CanonLog2 CinemaGamut D55"
    IDTs.append(CLog2_IDT)
    #CanonLinear_IDT = "Linear CinemaGamut D55"
    #IDTs.append(CanonLinear_IDT)
    CLog3_IDT = "CanonLog3 CinemaGamut D55"
    IDTs.append(CLog3_IDT)
    #VLin_IDT = "Linear V-Gamut"
    #IDTs.append(VLin_IDT)
    VLogVGamut_IDT = "V-Log V-Gamut"
    IDTs.append(VLogVGamut_IDT)
    #LinRWD_IDT = "Linear REDWideGamutRGB"
    #IDTs.append(LinRWD_IDT)
    Log3G10_IDT = "Log3G10 REDWideGamutRGB"
    IDTs.append(Log3G10_IDT)
    #LinSGamut3_IDT = "Linear S-Gamut3"
    #IDTs.append(LinSGamut3_IDT)
    SLOG_IDT = "S-Log3 S-Gamut3"
    IDTs.append(SLOG_IDT)
    #LinSGamut3cine_IDT = "Linear S-Gamut3.Cine"
    #IDTs.append(LinSGamut3cine_IDT)
#    SLOGcine_IDT = "S-Log3 S-Gamut3.Cine"
#    IDTs.append(SLOGcine_IDT)
    #LinVeniceSGamut3_IDT = "Linear Venice S-Gamut3"
    #IDTs.append(LinVeniceSGamut3_IDT)
    #SLog3Venice_IDT = "S-Log3 Venice S-Gamut3"
    #IDTs.append(SLog3Venice_IDT)
    #LinVeniceSGamut3cine_IDT = "Linear Venice S-Gamut3.Cine"
    #IDTs.append(LinVeniceSGamut3cine_IDT)
    #SLog3VeniceCine_IDT = "S-Log3 Venice S-Gamut3.Cine"
    #IDTs.append(SLog3VeniceCine_IDT)
#    Cam709_IDT = "Camera Rec.709"
#    IDTs.append(Cam709_IDT)
#    P3_IDT = "sRGB Encoded P3-D65"
#    IDTs.append(P3_IDT)
#    ACEScg_IDT = "ACEScg"
#    IDTs.append(ACEScg_IDT)
#    ACES2065_1_IDT = "ACES2065-1"
#    IDTs.append(ACES2065_1_IDT)
#    Rec1886_Rec709_IDT = "Rec.1886 Rec.709 - Display"
#    IDTs.append(Rec1886_Rec709_IDT)


    ## defining the ODT
    Rec709_ODT = "Rec.1886 Rec.709 - Display"
    P3D65_ODT = "P3-D65 - Display"


    ## defining the video to be imported **** TO BE CHANGED SO I CAN RUN THIS OFF COMMAND LINE*****
    Src_VideoPath = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\DJI_Test_Footage\ShortClip.mov"
    #Src_VideoPath = r"C:\Users\malco\Root\8_Special_Projects\ZS_Stills\A002_C005_0113GW.RDC\RWGLog3G10Clip.mov"
    #Src_VideoPath = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\DJI_Test_Footage\CowsSmall.mp4"
    #Src_VideoPath = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\Dev_ColorSpaceSniffer\001.TokyoNight_Cam1.mov"
    #Src_VideoPath = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\Slog3_Test_Footage\shortSlog.mov"
   # Src_VideoPath = r"C:\Users\malco\Root\999_misc\test_videos\BabyHands.mp4"



    SampleFrames = extract_frames(Src_VideoPath, 1)

    ArrayOfScores = []
    counter = 0


    #adding a bunch of bullshit to handle image saving in a way taht doesn't make me want to kms
    cwd = os.getcwd()
    outputDir = "output"
    outputPath = os.path.join(cwd, outputDir)
 #  create directory if needed
    os.makedirs(outputPath, exist_ok=True)
 

    print("beginning ananlysis of ", len(IDTs), "IDTs")


    for idt in IDTs :
        #print('running round number ', counter,'  of this loop')
        PenaltyScore = 0.0
        TransformedFrames = OCIO_CST(SampleFrames, config, idt, P3D65_ODT)
#        PenaltyScore = ScoreIDTtransforms(SampleFrames, TransformedFrames, config, idt, P3D65_ODT)
        ArrayOfScores.append(PenaltyScore)

        print("the Penalty score for the transform ", idt, " is  ", PenaltyScore)

        displayMe_clamp = np.clip(TransformedFrames[0], 0, 1.0)
        displayMe8bit = (displayMe_clamp * 255.0).astype(np.uint8)
        displayMeRGB = cv.cvtColor(displayMe8bit, cv.COLOR_RGB2BGR)
        dimensions = (1280, 720)
        #cv.imshow('this is the CST Frame', cv.resize(displayMeRGB, dimensions))

        outputName = os.path.join(outputPath, f"{idt}.png")

        ok = cv.imwrite(outputName, displayMeRGB)
        if not ok :
            print("FAILED TO WRITE:", outputName)
        counter += 1
        #clearing out the array

    min_pos = ArrayOfScores.index(min(ArrayOfScores))


    print("The Results are in, accordin to my studies, the least bad IDT for you to use is......", IDTs[min_pos])



    #Avg_SNR , Avg_Noise = CheckNoiseLevels(DJI2ODT_Frames)


    #print('The average number of clipped red pixels per frame is: ', AvgClippedPix[2])
    #print('The average number of clipped green pixels per frame is: ', AvgClippedPix[1])
    #print('The average number of clipped blue pixels per frame is: ', AvgClippedPix[0])
    #print('the average Noise in the Red Channel is: ', Avg_Noise[2])
    #print('the average Noise in the green Channel is: ', Avg_Noise[1])
    #print('the average Noise in the blue Channel is: ', Avg_Noise[0])
    #print('the average Noise in the Luma Channel is: ', Avg_Noise[2])
    #print('the average SNR is: ', Avg_SNR)


    #while True:
    #    k = cv.waitKey()
    #    if k == 27:
    #        break

    cv.destroyAllWindows

    fourcc = cv.VideoWriter_fourcc(*'FFV1')
    #output = cv.VideoWriter('IntermediateOutput.mov', fourcc, FrameRate, (int(FrameWidth), int(FrameHeight)))
    #output.release()

    
    cv.destroyAllWindows




if __name__ == "__main__" :
    main()
