import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.filters import frangi
from skimage import morphology
from scipy.ndimage import convolve

# --- WEB APP INTERFACE SETUP ---
st.set_page_config(page_title="CAM Vessel Analyzer Pro", layout="wide")
st.title("🔬 CAM Assay Blood Vessel Analyzer (Pro)")
st.write("A deterministic, mathematical pipeline for measuring angiogenesis and vessel hierarchy.")

# --- INTERACTIVE SIDEBAR CONTROLS ---
st.sidebar.header("📏 Calibration (Units)")
unit_choice = st.sidebar.selectbox("Measurement Unit", ["Pixels (px)", "Micrometers (µm)", "Millimeters (mm)", "Centimeters (cm)"])

if unit_choice == "Pixels (px)":
    scale_factor = 1.0
    u_label = "px"
else:
    u_label = unit_choice.split(" ")[1].replace("(", "").replace(")", "")
    st.sidebar.info(f"How many pixels equal 1 {u_label}? (Check your microscope scale bar)")
    known_px = st.sidebar.number_input("Pixels measured:", min_value=1.0, value=100.0, step=10.0)
    known_unit = st.sidebar.number_input(f"Equals how many {u_label}?", min_value=0.001, value=1.0, step=0.1)
    scale_factor = known_unit / known_px

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Algorithm Tuning")
auto_tune = st.sidebar.checkbox("🤖 Auto-Tune Parameters", value=True)

st.sidebar.write("**Manual Override**")
threshold_val = st.sidebar.slider("Vessel Strictness", 1, 50, 5, disabled=auto_tune)
blur_val = st.sidebar.slider("Cookie Cutter Blur", 11, 201, 41, step=2, disabled=auto_tune)

st.sidebar.markdown("---")
st.sidebar.write("**🩸 Vein Classification Tuning**")
primary_thresh = st.sidebar.slider("Primary Min Radius (px)", 1.0, 20.0, 3.0, step=0.5)
secondary_thresh = st.sidebar.slider("Secondary Min Radius (px)", 0.5, 10.0, 1.5, step=0.5)

def analyze_cam_vessels(img, auto, thresh_manual, blur_manual, p_thresh, s_thresh):
    green_channel = img[:, :, 1] 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(green_channel)
    
    if auto:
        calc_blur = int(img.shape[1] * 0.05)
        blur_size = calc_blur if calc_blur % 2 != 0 else calc_blur + 1
    else:
        blur_size = blur_manual

    blur = cv2.GaussianBlur(green_channel, (blur_size, blur_size), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    egg_mask = np.zeros_like(green_channel)
    embryo_diameter_px = 0
    embryo_area_px = 0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(egg_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        embryo_area_px = cv2.contourArea(largest_contour)
        embryo_diameter_px = 2 * np.sqrt(embryo_area_px / np.pi)
        kernel = np.ones((15, 15), np.uint8)
        egg_mask = cv2.erode(egg_mask, kernel, iterations=2)

    inverted_img = cv2.bitwise_not(enhanced_img)
    vessels = frangi(inverted_img, sigmas=range(1, 10, 2), black_ridges=False)
    vessels_norm = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    if auto:
        otsu_thresh, _ = cv2.threshold(vessels_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_thresh = max(1, int(otsu_thresh * 0.5))
    else:
        final_thresh = thresh_manual

    _, binary_mask = cv2.threshold(vessels_norm, final_thresh, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=egg_mask)
    
    total_area_px = np.sum(binary_mask > 0)
    vessel_density = (total_area_px / embryo_area_px * 100) if embryo_area_px > 0 else 0
    
    bool_mask = binary_mask > 0
    skeleton = morphology.skeletonize(bool_mask)
    
    kernel
