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
st.sidebar.header("⚙️ Algorithm Tuning")
auto_tune = st.sidebar.checkbox("🤖 Auto-Tune Parameters", value=True, help="Automatically calculates the best strictness based on lighting.")

st.sidebar.markdown("---")
st.sidebar.write("**Manual Override** (Uncheck Auto-Tune to use these)")
threshold_val = st.sidebar.slider("Vessel Strictness", 1, 50, 5, disabled=auto_tune)
blur_val = st.sidebar.slider("Cookie Cutter Blur", 11, 201, 41, step=2, disabled=auto_tune)

st.sidebar.markdown("---")
st.sidebar.write("**🩸 Vein Classification Tuning**")
st.sidebar.info("Adjust these to match your microscope's zoom level. Watch the 'Color-Coded Hierarchy' image update in real-time.")
primary_thresh = st.sidebar.slider("Primary Min Radius (px)", 1.0, 20.0, 3.0, step=0.5)
secondary_thresh = st.sidebar.slider("Secondary Min Radius (px)", 0.5, 10.0, 1.5, step=0.5)

def analyze_cam_vessels(img, auto, thresh_manual, blur_manual, p_thresh, s_thresh):
    # Extract green channel for best contrast
    green_channel = img[:, :, 1] 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(green_channel)
    
    # --- 1. EMBRYO DIAMETER & AREA (ROI MASK) ---
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

    # --- 2. VESSEL DETECTION ---
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
    
    # Calculate Advanced Metrics
    vessel_density = (total_area_px / embryo_area_px * 100) if embryo_area_px > 0 else 0
    
    # --- 3. SKELETONIZATION & BRANCHING ---
    bool_mask = binary_mask > 0
    skeleton = morphology.skeletonize(bool_mask)
    
    kernel_conv = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    skeleton_int = skeleton.astype(np.uint8)
    filtered = convolve(skeleton_int, kernel_conv, mode='constant')
    
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)
    total_length = np.sum(skeleton)
    
    avg_width = (total_area_px / total_length) if total_length > 0 else 0
    
    # --- 4. COLOR CODED VEIN HIERARCHY (Size & Visuals) ---
    # Distance transform measures the exact physical thickness (radius) of every vein
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Isolate veins based on the sliders
    primary_mask = (skeleton) & (dist_transform > p_thresh)
    secondary_mask = (skeleton) & (dist_transform > s_thresh) & (dist_transform <= p_thresh)
    tertiary_mask = (skeleton) & (dist_transform > 0) & (dist_transform <= s_thresh)
    
    primary_vessels = np.sum(primary_mask)
    secondary_vessels = np.sum(secondary_mask)
    tertiary_vessels = np.sum(tertiary_mask)
    
    # Visual Magnifier: Thicken the lines to 3px so they are visible on web browsers
    kernel_vis = np.ones((3,3), np.uint8)
    vis_primary = cv2.dilate(primary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    vis_secondary = cv2.dilate(secondary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    vis_tertiary = cv2.dilate(tertiary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    
    # Paint the RGB color map (Drawn in reverse order so Red arteries sit on top)
    color_map = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    color_map[vis_tertiary] = [50, 50, 255]   # Blue for Tertiary
    color_map[vis_secondary] = [50, 255, 50]  # Green for Secondary
    color_map[vis_primary] = [255, 50, 50]    # Red for Primary
    
    # Overlay Branch Points (Red Dots)
    img_overlay = img.copy()
    y, x = np.where(branch_points)
    for i in range(len(x)):
        cv2.circle(img_overlay, (x[i], y[i]), 3, (0, 0, 255), -1) 
        
    return {
        "thresh_used": final_thresh,
        "blur_used": blur_size,
        "embryo_diameter": int(embryo_diameter_px),
        "embryo_area": int(embryo_area_px),
        "total_area": int(total_area_px),
        "vessel_density": round(vessel_density, 2),
        "avg_width": round(avg_width, 2),
        "total_length": int(total_length),
        "branches": int(num_branches),
        "primary": int(primary_vessels),
        "secondary": int(secondary_vessels),
        "tertiary": int(tertiary_vessels),
        "color_map": color_map,
        "overlay_img": img_overlay,
        "mask_img": egg_mask,
        "skeleton_img": skeleton
    }

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🔬 Single Image Analysis", "⚖️ Compare Two Assays"])

with tab1:
    uploaded_file = st.file_uploader("Upload CAM Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Running Mathematical Pipeline...'):
            results = analyze_cam_vessels(image, auto_tune, threshold_val, blur_val, primary_thresh, secondary_thresh)
        
        st.success(f"Analysis Complete! (Used Threshold: {results['thresh_used']}, Blur: {results['blur_used']})")
        
        # --- NEW MASSIVE MACRO DETAILS SECTION ---
        st.subheader("📊 Detailed Macro Measurements")
        
        # Row 1: The Egg itself
        st.write("**Embryo / Membrane Constraints**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Embryo Area (px²)", f"{results['embryo_area']:,}")
        m2.metric("Embryo Diameter (px)", f"{results['embryo_diameter']:,}")
        m3.metric("Vascular Density", f"{results['vessel_density']}%", help="Percentage of the Embryo covered by blood vessels")
        
        st.write("---")
        
        # Row 2: The Vessels
        st.write("**Overall Angiogenesis**")
        m4, m5, m6, m7 = st.columns(4)
        m4.metric("Total Vessel Area (px²)", f"{results['total_area']:,}")
        m5.metric("Total Vessel Length (px)", f"{results['total_length']:,}")
        m6.metric("Branch Points", f"{results['branches']:,}")
        m7.metric("Avg. Vessel Width (px)", f"{results['avg_width']}", help="Total Area divided by Total Length")
        
        st.write("---")
        
        # Row 3: Hierarchy
        st.subheader("🩸 Vessel Hierarchy (Length by Size)")
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Total Length", f"{results['total_length']:,}")
        hc2.metric("🟥 Primary Veins", f"{results['primary']:,}")
        hc3.metric("🟩 Secondary Veins", f"{results['secondary']:,}")
        hc4.metric("🟦 Tertiary Veins", f"{results['tertiary']:,}")

        # --- CSV EXPORT ---
        df = pd.DataFrame([{
            "Image Name": uploaded_file.name,
            "Embryo Area (px)": results['embryo_area'],
            "Embryo Diameter (px)": results['embryo_diameter'],
            "Vascular Density (%)": results['vessel_density'],
            "Avg Vessel Width (px)": results['avg_width'],
            "Total Vessel Area (px)": results['total_area'],
            "Total Vessel Length (px)": results['total_length'],
            "Branch Intersections": results['branches'],
            "Primary Veins (Red)": results['primary'],
            "Secondary Veins (Green)": results['secondary'],
            "Tertiary Veins (Blue)": results['tertiary']
        }])
        st.download_button("💾 Download Detailed Data as CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=f"cam_analysis_detailed_{uploaded_file.name}.csv", mime="text/csv")
        
        # --- VISUAL RESULTS ---
        st.write("---")
        st.subheader("📷 Visual Results")
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(results['overlay_img'], cv2.COLOR_BGR2RGB)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.image(img_rgb, caption="Original", use_container_width=True)
        c2.image(results['mask_img'], caption="Detection Zone", use_container_width=True, clamp=True)
        c3.image(results['color_map'], caption="Color-Coded Hierarchy", use_container_width=True, clamp=True)
        c4.image(overlay_rgb, caption="Branch Points", use_container_width=True)

with tab2:
    st.write("Upload a Control egg and a Treated egg to see a direct data comparison.")
    col_a, col_b = st.columns(2)
    with col_a:
        egg_a = st.file_uploader("Upload Egg A (Control)", type=["jpg", "jpeg", "png"], key="egg_a")
    with col_b:
        egg_b = st.file_uploader("Upload Egg B (Treated)", type=["jpg", "jpeg", "png"], key="egg_b")

    if egg_a and egg_b:
        with st.spinner('Analyzing both assays...'):
            img_a = cv2.imdecode(np.asarray(bytearray(egg_a.read()), dtype=np.uint8), 1)
            img_b = cv2.imdecode(np.asarray(bytearray(egg_b.read()), dtype=np.uint8), 1)
            
            res_a = analyze_cam_vessels(img_a, auto_tune, threshold_val, blur_val, primary_thresh, secondary_thresh)
            res_b = analyze_cam_vessels(img_b, auto_tune, threshold_val, blur_val, primary_thresh, secondary_thresh)
            
        st.subheader("📊 Comparison Chart")
        
        # Updated comparison chart to include new metrics
        comp_df = pd.DataFrame({
            "Metric": ["Total Length", "Branch Points", "Vessel Density (%)", "Vessel Area", "Primary Veins", "Secondary Veins", "Tertiary Veins"],
            "Egg A (Control)": [res_a['total_length'], res_a['branches'], res_a['vessel_density'], res_a['total_area'], res_a['primary'], res_a['secondary'], res_a['tertiary']],
            "Egg B (Treated)": [res_b['total_length'], res_b['branches'], res_b['vessel_density'], res_b['total_area'], res_b['primary'], res_b['secondary'], res_b['tertiary']]
        }).set_index("Metric")
        
        st.bar_chart(comp_df, height=400)
