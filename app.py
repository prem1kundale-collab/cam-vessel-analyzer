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
st.write("Upload a CAM assay image and use the sidebar to tune the detection algorithm in real-time.")

# --- INTERACTIVE SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Algorithm Tuning")
st.sidebar.write("Adjust these dials until your skeleton mask looks perfect.")

# Dial 1: Fixes the Black Screen!
threshold_val = st.sidebar.slider(
    "Vessel Detection Strictness", 
    min_value=1, max_value=50, value=5, step=1,
    help="Lower this if the mask is black. Raise it if the mask looks like static noise."
)

# Dial 2: Fixes the Cookie Cutter!
blur_val = st.sidebar.slider(
    "Cookie Cutter Blur Size", 
    min_value=11, max_value=151, value=41, step=2,
    help="Controls how much of the egg is kept. Must be an odd number."
)

def analyze_cam_vessels(img, thresh_val, blur_size):
    green_channel = img[:, :, 1] 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(green_channel)
    
    # --- 1. EMBRYO DIAMETER (ROI MASK) ---
    blur = cv2.GaussianBlur(green_channel, (blur_size, blur_size), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    egg_mask = np.zeros_like(green_channel)
    embryo_diameter_px = 0
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(egg_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        area = cv2.contourArea(largest_contour)
        embryo_diameter_px = 2 * np.sqrt(area / np.pi)
        
        kernel = np.ones((15, 15), np.uint8)
        egg_mask = cv2.erode(egg_mask, kernel, iterations=2)

    # --- 2. VESSEL DETECTION ---
    inverted_img = cv2.bitwise_not(enhanced_img)
    vessels = frangi(inverted_img, sigmas=range(1, 10, 2), black_ridges=False)
    vessels_norm = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # *** HERE IS WHERE YOUR SLIDER CONNECTS TO THE MATH ***
    _, binary_mask = cv2.threshold(vessels_norm, thresh_val, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=egg_mask)
    
    total_area_px = np.sum(binary_mask > 0)
    
    # --- 3. SKELETONIZATION & BRANCHING ---
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
    
    # --- 4. VEIN HIERARCHY ---
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    vessel_radii = dist_transform[skeleton]
    
    primary_vessels = np.sum(vessel_radii > 3.0)
    secondary_vessels = np.sum((vessel_radii > 1.5) & (vessel_radii <= 3.0))
    tertiary_vessels = np.sum((vessel_radii > 0) & (vessel_radii <= 1.5))
    
    img_overlay = img.copy()
    y, x = np.where(branch_points)
    for i in range(len(x)):
        cv2.circle(img_overlay, (x[i], y[i]), 3, (0, 0, 255), -1) 
        
    return {
        "embryo_diameter": int(embryo_diameter_px),
        "total_area": int(total_area_px),
        "total_length": int(total_length),
        "branches": int(num_branches),
        "primary": int(primary_vessels),
        "secondary": int(secondary_vessels),
        "tertiary": int(tertiary_vessels),
        "skeleton_img": skeleton,
        "overlay_img": img_overlay,
        "mask_img": egg_mask
    }

# --- WEB APP LOGIC ---
uploaded_file = st.file_uploader("Upload Image File Manually (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    with st.spinner('Running Smart Analysis...'):
        # Pass the slider values into our function
        results = analyze_cam_vessels(image, threshold_val, blur_val)
    
    # --- VISUALS (Moved to the top so you can see changes instantly) ---
    st.write("---")
    st.subheader("📷 Live Tuning Results")
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(results['overlay_img'], cv2.COLOR_BGR2RGB)
    
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.image(img_rgb, caption="Original Uploaded Image", use_container_width=True)
        st.image(results['skeleton_img'], caption="Vessel Skeleton Mask", use_container_width=True, clamp=True)
        
    with img_col2:
        st.image(results['mask_img'], caption="Detection Zone (Cookie Cutter)", use_container_width=True, clamp=True)
        st.image(overlay_rgb, caption="Detected Branch Points (Red Dots)", use_container_width=True)

    # --- DISPLAY METRICS ---
    st.write("---")
    st.subheader("📊 Macro Measurements")
    col1, col2, col3 = st.columns(3)
    col1.metric("Embryo Diameter (px)", f"{results['embryo_diameter']:,}")
    col2.metric("Total Vessel Area (px)", f"{results['total_area']:,}")
    col3.metric("Detected Branch Points", f"{results['branches']:,}")
    
    st.subheader("🩸 Vessel Hierarchy (Length by Thickness)")
    hc1, hc2, hc3, hc4 = st.columns(4)
    hc1.metric("Total Length", f"{results['total_length']:,}")
    hc2.metric("Primary Veins", f"{results['primary']:,}")
    hc3.metric("Secondary Veins", f"{results['secondary']:,}")
    hc4.metric("Tertiary Veins", f"{results['tertiary']:,}")
    
    # --- CSV EXPORT ---
    st.write("---")
    st.subheader("💾 Export Results")
    
    df = pd.DataFrame([{
        "Image Name": uploaded_file.name,
        "Strictness Threshold Used": threshold_val,
        "Embryo Diameter (px)": results['embryo_diameter'],
        "Total Area (px)": results['total_area'],
        "Total Length (px)": results['total_length'],
        "Branch Intersections": results['branches'],
        "Primary Vein Length (px)": results['primary'],
        "Secondary Vein Length (px)": results['secondary'],
        "Tertiary Vein Length (px)": results['tertiary']
    }])
    
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"cam_analysis_{uploaded_file.name}.csv",
        mime="text/csv",
    )
