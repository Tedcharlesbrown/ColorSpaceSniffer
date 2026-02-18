
import cv2 as cv
import numpy as np
import PyOpenColorIO as OCIO
import argparse
import os
import csv
from processing import extract_frames, OCIO_CST, load_image, STILL_IMAGE_EXTENSIONS
from metrics import ScoreIDTtransforms, BlackPedestalCheck
from gamut_wall import gamut_wall_test

def main():
    parser = argparse.ArgumentParser(description='ColorSpaceSniffer — identify the best IDT for your footage')
    parser.add_argument('video', help='Path to source video file or still image (.tiff, .png, .exr, etc.)')
    parser.add_argument('--config', default=None, help='Path to OCIO config file (.ocio). Default: ACES Studio Config v4.0.0 (built-in)')
    parser.add_argument('--samples', type=int, default=1, help='Number of frames to sample (default: 1)')
    parser.add_argument('--output', default='output', help='Output directory for preview images (default: ./output)')
    args = parser.parse_args()

    ###Setting the OCIO Config
    if args.config:
        config = OCIO.Config.CreateFromFile(args.config)
        print(f"Using OCIO config: {args.config}")
    else:
        config = OCIO.Config.CreateFromBuiltinConfig('studio-config-v4.0.0_aces-v2.0_ocio-v2.5')
        print("Using built-in ACES Studio Config v4.0.0 (ACES v2.0, OCIO v2.5)")

    ###Setting the OCIO transforms to be appraised during this appraisal
    IDTs = [
        # DJI
        "D-Log D-Gamut",
        # Apple
        "Apple Log",
        # ARRI
        "ARRI LogC3 (EI800)",
        "ARRI LogC4",
        # Blackmagic
        "BMDFilm WideGamut Gen5",
        "DaVinci Intermediate WideGamut",
        # Canon
        "CanonLog2 CinemaGamut D55",
        "CanonLog3 CinemaGamut D55",
        # Panasonic
        "V-Log V-Gamut",
        # RED
        "Log3G10 REDWideGamutRGB",
        # Sony
        "S-Log3 S-Gamut3",
        "S-Log3 S-Gamut3.Cine",
        "S-Log3 Venice S-Gamut3",
        "S-Log3 Venice S-Gamut3.Cine",
    ]

    ## defining the ODT
    Rec709_ODT = "Rec.1886 Rec.709 - Display"
    P3D65_ODT = "P3-D65 - Display"

    Src_VideoPath = args.video
    if Src_VideoPath.lower().endswith(STILL_IMAGE_EXTENSIONS):
        print(f"Still image detected — loading directly: {os.path.basename(Src_VideoPath)}")
        SampleFrames = load_image(Src_VideoPath)
    else:
        SampleFrames = extract_frames(Src_VideoPath, args.samples)

    ArrayOfScores = []
    counter = 0

    outputPath = args.output
    os.makedirs(outputPath, exist_ok=True)

    # --- Stage 1: Gamut Wall Test (Log vs Display-Referred Detection) ---
    gw_result = gamut_wall_test(SampleFrames, verbose=True, output_path=outputPath,
                                 image_name=os.path.splitext(os.path.basename(Src_VideoPath))[0])
    
    if gw_result['is_display_referred']:
        print(f"\n{'='*60}")
        print(f"  ⚠️  STAGE 1 RESULT: Footage appears DISPLAY-REFERRED")
        print(f"  Confidence: {gw_result['confidence']:.2f}")
        print(f"  Log IDT scoring will proceed, but results may be unreliable.")
        print(f"  Consider using an Inverse Tone Map (ITM) instead of a log IDT.")
        print(f"{'='*60}")
    else:
        print(f"\n  ✅ Stage 1: Footage appears to be log-encoded (confidence: {1.0 - gw_result['confidence']:.2f})")

    # --- Black Pedestal Diagnostic (DISABLED — too aggressive, eliminates 13/14 on real footage) ---
    darkest_neutral = None

    print(f"\n--- Scoring {len(IDTs)} IDTs ---")

    AllBreakdowns = []

    for idt in IDTs :
        #print('running round number ', counter,'  of this loop')
        PenaltyScore = 0.0
        TransformedFrames = OCIO_CST(SampleFrames, config, idt, P3D65_ODT)
        PenaltyScore, breakdown = ScoreIDTtransforms(SampleFrames, TransformedFrames, config, idt, P3D65_ODT,
                                                      darkest_neutral_luma=darkest_neutral)
        ArrayOfScores.append(PenaltyScore)
        AllBreakdowns.append(breakdown)

        print("the Penalty score for the transform ", idt, " is  ", PenaltyScore)

        displayMe_clamp = np.clip(TransformedFrames[0], 0, 1.0)
        displayMe8bit = (displayMe_clamp * 255.0).astype(np.uint8)
        displayMeRGB = cv.cvtColor(displayMe8bit, cv.COLOR_RGB2BGR)
        dimensions = (1280, 720)
        #cv.imshow('this is the CST Frame', cv.resize(displayMeRGB, dimensions))

        # Stamp IDT name and score onto the image
        score_text = f"IDT: {idt}   Score: {PenaltyScore:.4f} (unscored)" if PenaltyScore == 0.0 else f"IDT: {idt}   Score: {PenaltyScore:.4f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = min(displayMeRGB.shape[1], displayMeRGB.shape[0]) / 1500.0
        thickness = max(1, int(font_scale * 2))
        padding = int(font_scale * 30) + 10
        # Shadow for readability
        cv.putText(displayMeRGB, score_text, (padding + 2, padding + 2), font, font_scale, (0, 0, 0), thickness + 2, cv.LINE_AA)
        # White text on top
        cv.putText(displayMeRGB, score_text, (padding, padding), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

        outputName = os.path.join(outputPath, f"{idt}.png")

        ok = cv.imwrite(outputName, displayMeRGB)
        if not ok :
            print("FAILED TO WRITE:", outputName)
        counter += 1

    min_pos = ArrayOfScores.index(min(ArrayOfScores))

    print("The Results are in, accordin to my studies, the least bad IDT for you to use is......", IDTs[min_pos])

    # Write detailed score breakdown to CSV
    reportPath = os.path.join(outputPath, "score_report.csv")
    if AllBreakdowns:
        fieldnames = list(AllBreakdowns[0].keys())
        with open(reportPath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for bd in sorted(AllBreakdowns, key=lambda x: x['FinalScore']):
                writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in bd.items()})
        print(f"\nDetailed score report written to: {reportPath}")


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
