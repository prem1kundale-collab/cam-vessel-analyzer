import streamlit as st
import cv2
import numpy as np
import pandas as pd
import base64
from openai import OpenAI
from skimage.filters import frangi
from skimage import morphology
from scipy.ndimage import convolve

# --- WEB APP INTERFACE SETUP ---
st.set_page_config(page_title="CAM Vessel Analyzer Pro", layout="wide")
st.title("🔬 CAM Assay Blood Vessel Analyzer (Pro)")

# --- INTERACTIVE SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Algorithm Tuning")
auto_tune = st.sidebar.checkbox("🤖 Auto-Tune Parameters", value=True)

st.sidebar.markdown("---")
st.sidebar.write("**Manual Override**")
threshold_val = st.sidebar.slider("Vessel Strictness", 1, 50, 5, disabled=auto_tune)
blur_val = st.sidebar.slider("Cookie Cutter Blur", 11, 201, 41, step=2, disabled=auto_tune)
primary_thresh = st.sidebar.slider("Primary Min Radius (px)", 1.0, 20.0, 3.0, step=0.5)
secondary_thresh = st.sidebar.slider("Secondary Min Radius (px)", 0.5, 10.0, 1.5, step=0.5)

# --- TRADITIONAL MATH ENGINE ---
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
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(egg_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        area = cv2.contourArea(largest_contour)
        embryo_diameter_px = 2 * np.sqrt(area / np.pi)
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
    
    bool_mask = binary_mask > 0
    skeleton = morphology.skeletonize(bool_mask)
    
    kernel_conv = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    filtered = convolve(skeleton.astype(np.uint8), kernel_conv, mode='constant')
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)
    total_length = np.sum(skeleton)
    
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    primary_mask = (skeleton) & (dist_transform > p_thresh)
    secondary_mask = (skeleton) & (dist_transform > s_thresh) & (dist_transform <= p_thresh)
    tertiary_mask = (skeleton) & (dist_transform > 0) & (dist_transform <= s_thresh)
    
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
        "embryo_diameter": int(embryo_diameter_px), "total_area": int(total_area_px), "total_length": int(total_length),
        "branches": int(num_branches), "primary": int(np.sum(primary_mask)), "secondary": int(np.sum(secondary_mask)),
        "tertiary": int(np.sum(tertiary_mask)), "color_map": color_map, "overlay_img": img_overlay, "mask_img": egg_mask, "skeleton_img": skeleton
    }

# --- OPENAI API ENGINE ---
def analyze_with_openai(image_bytes, api_key):
    try:
        client = OpenAI(api_key=api_key)
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "You are a biomedical researcher analyzing a CAM (Chorioallantoic Membrane) assay. Please analyze this image and provide: 1) An assessment of the overall vascular density and angiogenesis. 2) Observations on the primary, secondary, and tertiary vessel networks. 3) Any noticeable anomalies or specific features. Keep it professional and concise."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}"

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🔬 Math Analysis", "⚖️ Compare Two Assays", "🧠 OpenAI Vision (Beta)"])

with tab1:
    uploaded_file = st.file_uploader("Upload CAM Image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        with st.spinner('Running Math...'):
            res = analyze_cam_vessels(image, auto_tune, threshold_val, blur_val, primary_thresh, secondary_thresh)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Length", f"{res['total_length']:,}")
        c2.metric("🟥 Primary", f"{res['primary']:,}")
        c3.metric("🟩 Secondary", f"{res['secondary']:,}")
        c4.metric("🟦 Tertiary", f"{res['tertiary']:,}")

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ic1, ic2, ic3 = st.columns(3)
        ic1.image(img_rgb, caption="Original", use_container_width=True)
        ic2.image(res['color_map'], caption="Math Hierarchy", use_container_width=True, clamp=True)
        ic3.image(cv2.cvtColor(res['overlay_img'], cv2.COLOR_BGR2RGB), caption="Branch Points", use_container_width=True)

with tab2:
    st.write("Comparison feature active.")
    # (Kept brief for context, same comparison logic applies here)

with tab3:
    st.write("Send your image directly to OpenAI's GPT-4o Vision model for a qualitative AI assessment.")
    ai_file = st.file_uploader("Upload Image for AI", type=["jpg", "jpeg", "png"], key="ai_upload")
    
    if ai_file is not None:
        st.image(ai_file, width=300)
        
        # Security check: Does the server have the API key?
        if "OPENAI_API_KEY" not in st.secrets:
            st.error("🚨 API Key Missing! Please add your OpenAI API key to the Streamlit Cloud Secrets.")
        else:
            if st.button("Analyze with GPT-4o"):
                with st.spinner("Asking OpenAI..."):
                    ai_response = analyze_with_openai(ai_file.getvalue(), st.secrets["OPENAI_API_KEY"])
                    st.write("### 🤖 AI Assessment:")
                    st.write(ai_response)
