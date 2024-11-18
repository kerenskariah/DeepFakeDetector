import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tempfile import TemporaryDirectory

# Function to extract frames from a video
def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 1
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        frame_count += 1
    cap.release()
    return frames  # Return list of extracted frame paths

# Function to calculate Mean Squared Error (MSE)
def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

# Function to compare images using MSE and SSIM
from skimage.metrics import structural_similarity as ssim

# Function to compare images with dynamic win_size for SSIM
def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    
    # Determine the smaller dimension to set an appropriate win_size
    min_dim = min(imageA.shape[0], imageA.shape[1])
    win_size = min(7, min_dim) if min_dim >= 7 else min_dim  # Set win_size to 3 if images are very small
    
    # Calculate SSIM with specified win_size and channel_axis
    s = ssim(imageA, imageB, win_size=win_size, channel_axis=2)
    return m, s


# Streamlit app interface
st.title("Deepfake Detection Tool")
st.write("Upload real and fake videos to extract and compare frames.")

# Upload files
real_video = st.file_uploader("Upload Real Video", type=["mp4", "avi", "mov"])
fake_video = st.file_uploader("Upload Fake Video", type=["mp4", "avi", "mov"])

if real_video and fake_video:
    with TemporaryDirectory() as temp_dir:
        real_video_path = os.path.join(temp_dir, "real_video.mp4")
        fake_video_path = os.path.join(temp_dir, "fake_video.mp4")
        
        # Save uploaded videos to temporary directory
        with open(real_video_path, 'wb') as f:
            f.write(real_video.read())
        with open(fake_video_path, 'wb') as f:
            f.write(fake_video.read())
        
        # Directories for extracted frames
        real_frame_dir = os.path.join(temp_dir, "real_frames")
        fake_frame_dir = os.path.join(temp_dir, "fake_frames")
        
        # Extract frames
        real_frames = extract_frames(real_video_path, real_frame_dir)
        fake_frames = extract_frames(fake_video_path, fake_frame_dir)
        
        if real_frames and fake_frames:
            # Load the first frame from each video for comparison
            imageA = cv2.imread(real_frames[0])
            imageB = cv2.imread(fake_frames[0])
            
            # Ensure images loaded correctly
            if imageA is None or imageB is None:
                st.error("Failed to load frames from one or both videos.")
            else:
                # Compare images
                mse_value, ssim_value = compare_images(imageA, imageB)
                
                # Display results
                st.write(f"**Mean Squared Error (MSE):** {mse_value:.2f}")
                st.write(f"**Structural Similarity Index (SSIM):** {ssim_value:.2f}")
                
                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(imageA, caption="Real Video Frame")
                with col2:
                    st.image(imageB, caption="Fake Video Frame")
                
                # Display comparison visualization
                st.write("**Comparison Visualization**")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                ax1.imshow(cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB))
                ax1.set_title("Real Frame")
                ax1.axis("off")
                ax2.imshow(cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB))
                ax2.set_title("Fake Frame")
                ax2.axis("off")
                st.pyplot(fig)
        else:
            st.error("Failed to extract frames from one or both videos.")
