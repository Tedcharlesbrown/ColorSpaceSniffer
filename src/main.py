
import cv2 as cv
import numpy as np
import PyOpenColorIO as OCIO
import argparse
import os
from processing import extract_frames, OCIO_CST
# from metrics import ScoreIDTtransforms # Uncomment when ready to use scoring

def main():
    # FIX: replaced hardcoded Windows paths with CLI arguments
    parser = argparse.ArgumentParser(description='ColorSpaceSniffer — identify the best IDT for your footage')
    parser.add_argument('video', help='Path to source video file')
    parser.add_argument('config', help='Path to OCIO config file (.ocio)')
    parser.add_argument('--samples', type=int, default=1, help='Number of frames to sample (default: 1)')
    parser.add_argument('--output', default='output', help='Output directory for preview images (default: ./output)')
    args = parser.parse_args()

    ###Setting the OCIO Config 
    config = OCIO.Config.CreateFromFile(args.config)

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

    Src_VideoPath = args.video
    SampleFrames = extract_frames(Src_VideoPath, args.samples)

    ArrayOfScores = []
    counter = 0

    outputPath = args.output
    os.makedirs(outputPath, exist_ok=True)

    print("beginning analysis of ", len(IDTs), "IDTs")

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
