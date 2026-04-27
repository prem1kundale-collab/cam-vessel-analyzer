import streamlit as st
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import morphology
from scipy.ndimage import convolve

# --- WEB APP INTERFACE SETUP ---
st.set_page_config(page_title="CAM Vessel Analyzer", layout="wide")

st.error("🚨 APP IS SUCCESSFULLY CONNECTED AND RUNNING! 🚨")

st.title("🔬 CAM Assay Blood Vessel Analyzer")
st.write("Upload a CAM assay image to automatically quantify blood vessel length and branch points.")

def analyze_cam_vessels(img):
    green_channel = img[:, :, 1] 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(green_channel)
    
    blur = cv2.GaussianBlur(green_channel, (41, 41), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    egg_mask = np.zeros_like(green_channel)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(egg_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((15, 15), np.uint8)
        egg_mask = cv2.erode(egg_mask, kernel, iterations=2)

    inverted_img = cv2.bitwise_not(enhanced_img)
    vessels = frangi(inverted_img, sigmas=range(1, 10, 2), black_ridges=False)
    vessels_norm = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    _, binary_mask = cv2.threshold(vessels_norm, 15, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=egg_mask)
    
    bool_mask = binary_mask > 0
    skeleton = morphology.skeletonize(bool_mask)
    
    kernel_conv = np.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])
    skeleton_int = skeleton.astype(np.uint8)
    filtered = convolve(skeleton_int, kernel_conv, mode='constant')
    
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)
    total_length = np.sum(skeleton)
    
    img_overlay = img.copy()
    y, x = np.where(branch_points)
    for i in range(len(x)):
        cv2.circle(img_overlay, (x[i], y[i]), 3, (0, 0, 255), -1) 
        
    return total_length, num_branches, skeleton, img_overlay, egg_mask

# --- WEB APP LOGIC ---
uploaded_file = st.file_uploader("Choose a CAM image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('Analyzing vessels... This may take a few seconds.'):
        length, branches, skeleton_img, overlay_img, mask_img = analyze_cam_vessels(image)
    
    st.success("Analysis Complete!")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Vessel Length (pixels)", f"{length:,}")
    col2.metric("Detected Branch Points", f"{branches:,}")
    
    st.write("---")
    st.subheader("Image Analysis Results")
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)
        st.image(skeleton_img, caption="Vessel Skeleton Mask", use_container_width=True, clamp=True)
        
    with img_col2:
        st.image(mask_img, caption="Detection Zone (Cookie Cutter)", use_container_width=True, clamp=True)
        st.image(overlay_rgb, caption="Detected Branch Points (Red Dots)", use_container_width=True)
