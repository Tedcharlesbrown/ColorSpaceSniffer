
import cv2 as cv
import numpy as np
import subprocess
import PyOpenColorIO as OCIO
import os

STILL_IMAGE_EXTENSIONS = ('.tiff', '.tif', '.png', '.exr', '.jpg', '.jpeg', '.dpx')

def load_image(ImagePath):
    """Load a single still image as a normalized float32 frame. Supports 8-bit and 16-bit."""
    raw = cv.imread(ImagePath, cv.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read image: {ImagePath}")

    # Convert to float32 and normalize based on bit depth
    if raw.dtype == np.uint8:
        frame = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        frame = raw.astype(np.float32) / 65535.0
    else:
        frame = raw.astype(np.float32)

    # Ensure 3 channels, then convert BGR→RGB to match extract_frames()
    if len(frame.shape) == 2:
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv.cvtColor(frame[:, :, :3], cv.COLOR_BGR2RGB)
    else:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    return [frame]  # return as list to match video frame array interface

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

        Frames.append(frame_rgb)

    else :
        for sample in range(SampleCount) :
            # FIX: was `Interval * sample - 1` which evaluates to -1 for sample=0,
            # causing OpenCV to read the last frame as the first sample.
            VideoFile.set(cv.CAP_PROP_POS_FRAMES, int(Interval * sample))
            ret , rawFrame = VideoFile.read()

            frame_rgb = cv.cvtColor(rawFrame, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

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


    OutputDir = os.path.join(os.getcwd(), "output")
    os.makedirs(OutputDir, exist_ok=True)

    #calling FFMPEG to extract N number of frames from the video at the desired interval at 16 bit depth
    for frame in range(SampleCount):
        ThisFrame = int(Interval * frame)
        ThisFrameFlag = f"select=eq(n\\,{ThisFrame})"
        outputFileName = os.path.join(OutputDir, f'IntermediateFrame{frame}.tiff')
        subprocess.run([
            'ffmpeg', '-y',
            '-i', VideoPath, '-map', '0:v:0', '-vf', ThisFrameFlag, '-vframes', '1',
            '-c:v', 'tiff', '-pix_fmt', 'rgb48le', outputFileName
        ])

    #reading in all of the intermediate frames, converting them to RGB, and then normalizing them
    for frame in range(SampleCount):
        inputFileName = os.path.join(OutputDir, f'IntermediateFrame{frame}.tiff')
        uint16Frame = cv.imread(inputFileName, cv.IMREAD_UNCHANGED)
        float_frame = uint16Frame.astype(np.float32) / (pow(2, 16) - 1)
        Frames.append(float_frame)
        os.remove(inputFileName)

    # Clean up directory if empty
    try:
        os.rmdir(OutputDir)
    except OSError:
        pass # Directory not empty, leave it
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
