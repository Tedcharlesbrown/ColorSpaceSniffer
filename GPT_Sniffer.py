import cv2
import numpy as np
import PyOpenColorIO as ocio
import subprocess
import os

def extract_frames(video_path, output_dir=None, max_frames=None):
    """
    Extract frames from a video file using OpenCV.
    Returns a list of frames as numpy arrays in float32 [0,1].
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB and normalize to [0,1]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame_rgb)

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

        # Optional: save frames to disk
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:05d}.png"),
                        cv2.cvtColor((frame_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    cap.release()
    return frames

def apply_ocio_transform(frames, config_path, input_space, output_space):
    """
    Apply an OCIO color transform to a list of frames.
    frames: list of float32 numpy arrays [0,1]
    config_path: path to OCIO config
    input_space: OCIO input color space name
    output_space: OCIO output color space name
    Returns list of transformed frames (float32 [0,1])
    """
    config = ocio.Config.CreateFromFile(config_path)
    processor = config.getProcessor(input_space, output_space)
    cpu_processor = processor.getDefaultCPUProcessor()

    transformed_frames = []
    for frame in frames:
        # PyOpenColorIO expects NxMx3 float32 array
        transformed = cpu_processor.applyRGB(frame)
        # Clip to [0,1] just in case
        #np.clip(frame, 0.0, 1.0)
        transformed_frames.append(frame)
    return transformed_frames

def main():
    video_file = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\Slog3_Test_Footage\shortSlog.mov"
    ocio_config = r"C:\Users\malco\Root\8_Special_Projects\COlorPipelineTest\Dev_ColorSpaceSniffer\ZaleTestOcioConfig_Dev.ocio"
    input_space = "S-Log3 S-Gamut3"
    output_space = "Rec.1886 Rec.709 - Display"

    # Extract frames
    frames = extract_frames(video_file, max_frames=2)  # for testing, limit frames

    # Apply OCIO transform
    transformed_frames = apply_ocio_transform(frames, ocio_config, input_space, output_space)

    # Optional: save transformed frames
    for i, frame in enumerate(transformed_frames):
        frame_clamp = np.clip(frame, 0, 1.0)
        frame_8bit = (frame_clamp * 255.0).astype(np.uint8)
        cv2.imwrite(f"transformed_frame_{i:04d}.png",
                    cv2.cvtColor(frame_8bit, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
