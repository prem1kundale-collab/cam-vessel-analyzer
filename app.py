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

# --- SYNCED SLIDERS & NUMBER BOXES ---
st.sidebar.markdown("---")
st.sidebar.write("**🩸 Vein Classification Tuning**")

# 1. Initialize memory banks for the numbers
if 'p_thresh' not in st.session_state:
    st.session_state.p_thresh = 3.0
if 's_thresh' not in st.session_state:
    st.session_state.s_thresh = 1.5

# 2. Callback functions to sync the widgets
def update_p_from_slider(): st.session_state.p_thresh = st.session_state.p_sl
def update_p_from_num(): st.session_state.p_thresh = st.session_state.p_nm
def update_s_from_slider(): st.session_state.s_thresh = st.session_state.s_sl
def update_s_from_num(): st.session_state.s_thresh = st.session_state.s_nm

# 3. Draw the Primary UI (Slider + Box)
st.sidebar.caption("🟥 Primary Min Radius (px)")
pc1, pc2 = st.sidebar.columns([3, 2])
pc1.slider("P_Slider", 1.0, 20.0, step=0.5, key="p_sl", value=st.session_state.p_thresh, on_change=update_p_from_slider, label_visibility="collapsed")
pc2.number_input("P_Num", 0.1, 500.0, step=0.5, key="p_nm", value=st.session_state.p_thresh, on_change=update_p_from_num, label_visibility="collapsed")

# 4. Draw the Secondary UI (Slider + Box)
st.sidebar.caption("🟩 Secondary Min Radius (px)")
sc1, sc2 = st.sidebar.columns([3, 2])
sc1.slider("S_Slider", 0.5, 10.0, step=0.5, key="s_sl", value=st.session_state.s_thresh, on_change=update_s_from_slider, label_visibility="collapsed")
sc2.number_input("S_Num", 0.1, 500.0, step=0.5, key="s_nm", value=st.session_state.s_thresh, on_change=update_s_from_num, label_visibility="collapsed")

# Assign the final synced values to our math variables
primary_thresh = st.session_state.p_thresh
secondary_thresh = st.session_state.s_thresh

# --- CORE MATH ENGINE ---
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
    
    kernel_conv = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    skeleton_int = skeleton.astype(np.uint8)
    filtered = convolve(skeleton_int, kernel_conv, mode='constant')
    
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)
    total_length = np.sum(skeleton)
    avg_width = (total_area_px / total_length) if total_length > 0 else 0
    
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    primary_mask = (skeleton) & (dist_transform > p_thresh)
    secondary_mask = (skeleton) & (dist_transform > s_thresh) & (dist_transform <= p_thresh)
    tertiary_mask = (skeleton) & (dist_transform > 0) & (dist_transform <= s_thresh)
    
    primary_length_px = np.sum(primary_mask)
    secondary_length_px = np.sum(secondary_mask)
    tertiary_length_px = np.sum(tertiary_mask)
    
    num_p, _ = cv2.connectedComponents((primary_mask.astype(np.uint8) * 255))
    num_s, _ = cv2.connectedComponents((secondary_mask.astype(np.uint8) * 255))
    num_t, _ = cv2.connectedComponents((tertiary_mask.astype(np.uint8) * 255))
    
    primary_count = max(0, num_p - 1)
    secondary_count = max(0, num_s - 1)
    tertiary_count = max(0, num_t - 1)
    
    kernel_vis = np.ones((3,3), np.uint8)
    vis_primary = cv2.dilate(primary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    vis_secondary = cv2.dilate(secondary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    vis_tertiary = cv2.dilate(tertiary_mask.astype(np.uint8), kernel_vis, iterations=1) > 0
    
    color_map = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
    color_map[vis_tertiary] = [50, 50, 255]   
    color_map[vis_secondary] = [50, 255, 50]  
    color_map[vis_primary] = [255, 50, 50]    
    
    img_overlay = img.copy()
    y, x = np.where(branch_points)
    for i in range(len(x)):
        cv2.circle(img_overlay, (x[i], y[i]), 3, (0, 0, 255), -1) 
        
    return {
        "thresh_used": final_thresh, "blur_used": blur_size,
        "embryo_diameter": embryo_diameter_px * scale_factor,
        "embryo_area": embryo_area_px * (scale_factor ** 2), 
        "total_area": total_area_px * (scale_factor ** 2),
        "vessel_density": vessel_density, 
        "avg_width": avg_width * scale_factor,
        "total_length": total_length * scale_factor,
        "branches": int(num_branches), 
        "primary_len": primary_length_px * scale_factor,
        "secondary_len": secondary_length_px * scale_factor,
        "tertiary_len": tertiary_length_px * scale_factor,
        "primary_count": primary_count,
        "secondary_count": secondary_count,
        "tertiary_count": tertiary_count,
        "color_map": color_map, "overlay_img": img_overlay, "mask_img": egg_mask, "skeleton_img": skeleton
    }

# --- TABS AND UPLOAD MENUS LAYOUT ---
tab1, tab2 = st.tabs(["🔬 Single Image Analysis", "⚖️ Compare Two Assays"])

with tab1:
    st.write("Upload a single CAM assay image for detailed calibrated analysis.")
    uploaded_file = st.file_uploader("Upload CAM Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single")

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        with st.spinner('Running Mathematical Pipeline...'):
            results = analyze_cam_vessels(image, auto_tune, threshold_val, blur_val, primary_thresh, secondary_thresh)
        
        st.subheader("📊 Detailed Macro Measurements")
        
        st.write("**Embryo / Membrane Constraints**")
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Embryo Area ({u_label}²)", f"{results['embryo_area']:,.2f}")
        m2.metric(f"Embryo Diameter ({u_label})", f"{results['embryo_diameter']:,.2f}")
        m3.metric("Vascular Density", f"{results['vessel_density']:,.2f}%")
        
        st.write("---")
        st.write("**Overall Angiogenesis**")
        m4, m5, m6, m7 = st.columns(4)
        m4.metric(f"Total Vessel Area ({u_label}²)", f"{results['total_area']:,.2f}")
        m5.metric(f"Total Vessel Length ({u_label})", f"{results['total_length']:,.2f}")
        m6.metric("Branch Points (Count)", f"{results['branches']:,}")
        m7.metric(f"Avg. Vessel Width ({u_label})", f"{results['avg_width']:,.2f}")
        
        st.write("---")
        st.subheader(f"🩸 Vessel Hierarchy (Lengths vs. Counts)")
        
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric(f"Total Length ({u_label})", f"{results['total_length']:,.2f}")
        hc2.metric(f"🟥 Primary Length", f"{results['primary_len']:,.2f}")
        hc3.metric(f"🟩 Secondary Length", f"{results['secondary_len']:,.2f}")
        hc4.metric(f"🟦 Tertiary Length", f"{results['tertiary_len']:,.2f}")
        
        cc1, cc2, cc3, cc4 = st.columns(4)
        total_segments = results['primary_count'] + results['secondary_count'] + results['tertiary_count']
        cc1.metric("Total Segments (Count)", f"{total_segments:,}")
        cc2.metric("🟥 Primary Count", f"{results['primary_count']:,}")
        cc3.metric("🟩 Secondary Count", f"{results['secondary_count']:,}")
        cc4.metric("🟦 Tertiary Count", f"{results['tertiary_count']:,}")

        df = pd.DataFrame([{
            "Image Name": uploaded_file.name,
            f"Embryo Area ({u_label}^2)": results['embryo_area'],
            f"Embryo Diameter ({u_label})": results['embryo_diameter'],
            "Vascular Density (%)": results['vessel_density'],
            f"Avg Vessel Width ({u_label})": results['avg_width'],
            f"Total Vessel Area ({u_label}^2)": results['total_area'],
            f"Total Vessel Length ({u_label})": results['total_length'],
            "Branch Intersections": results['branches'],
            f"Primary Length ({u_label})": results['primary_len'],
            f"Secondary Length ({u_label})": results['secondary_len'],
            f"Tertiary Length ({u_label})": results['tertiary_len'],
            "Primary Veins (Count)": results['primary_count'],
            "Secondary Veins (Count)": results['secondary_count'],
            "Tertiary Veins (Count)": results['tertiary_count']
        }])
        st.download_button("💾 Download Detailed Data as CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=f"cam_analysis_calibrated_{uploaded_file.name}.csv", mime="text/csv")
        
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
            
        st.subheader(f"📊 Comparison Chart")
        
        comp_df = pd.DataFrame({
            "Metric": [
                f"Total Length ({u_label})", "Branch Points", "Vessel Density (%)", 
                f"Primary Length ({u_label})", "Primary Count", 
                f"Secondary Length ({u_label})", "Secondary Count"
            ],
            "Egg A (Control)": [
                res_a['total_length'], res_a['branches'], res_a['vessel_density'], 
                res_a['primary_len'], res_a['primary_count'], 
                res_a['secondary_len'], res_a['secondary_count']
            ],
            "Egg B (Treated)": [
                res_b['total_length'], res_b['branches'], res_b['vessel_density'], 
                res_b['primary_len'], res_b['primary_count'], 
                res_b['secondary_len'], res_b['secondary_count']
            ]
        }).set_index("Metric")
        
        st.bar_chart(comp_df, height=500)
