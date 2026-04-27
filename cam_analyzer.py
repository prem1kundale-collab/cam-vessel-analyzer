import cv2
import numpy as np
from skimage.filters import frangi
from skimage import morphology
from scipy.ndimage import convolve
import matplotlib.pyplot as plt

def analyze_cam_vessels(image_path):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Image not found. Check the path."
    
    green_channel = img[:, :, 1] 
    
    # 2. Enhance Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(green_channel)
    
    # --- NEW STEP: THE DIGITAL COOKIE CUTTER (ROI MASK) ---
    # We blur the image heavily so tiny vessels disappear, leaving only the big egg shape
    blur = cv2.GaussianBlur(green_channel, (41, 41), 0)
    
    # We use a threshold to separate the main bright area (egg) from the darker background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find the outlines (contours) of all shapes in this basic black/white image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank black canvas for our mask
    egg_mask = np.zeros_like(green_channel)
    
    if contours:
        # We assume the physically largest shape on screen is your egg/assay!
        largest_contour = max(contours, key=cv2.contourArea)
        # Draw that shape filled with pure white
        cv2.drawContours(egg_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Shrink (erode) the mask boundary inward slightly. 
        # This prevents the algorithm from accidentally counting the edges of the petri dish/eggshell!
        kernel = np.ones((15, 15), np.uint8)
        egg_mask = cv2.erode(egg_mask, kernel, iterations=2)
    # -----------------------------------------------------

    # 3. Apply the Frangi Filter
    inverted_img = cv2.bitwise_not(enhanced_img)
    vessels = frangi(inverted_img, sigmas=range(1, 10, 2), black_ridges=False)
    vessels_norm = cv2.normalize(vessels, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 4. Thresholding
    _, binary_mask = cv2.threshold(vessels_norm, 15, 255, cv2.THRESH_BINARY)
    
    # --- APPLY THE MASK ---
    # We overlay the cookie cutter! Any vessel detected outside the white egg_mask is deleted.
    binary_mask = cv2.bitwise_and(binary_mask, binary_mask, mask=egg_mask)
    
    # 5. Skeletonization
    bool_mask = binary_mask > 0
    skeleton = morphology.skeletonize(bool_mask)
    
    # 6. Quantification: Counting Branch Points
    kernel_conv = np.array([[1, 1, 1],
                            [1, 10, 1],
                            [1, 1, 1]])
    
    skeleton_int = skeleton.astype(np.uint8)
    filtered = convolve(skeleton_int, kernel_conv, mode='constant')
    
    branch_points = filtered > 12
    num_branches = np.sum(branch_points)
    total_length = np.sum(skeleton)
    
    # --- 7. Visualizing the Results ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original CAM Image')
    
    # We will show the Mask here instead so you can verify it's cutting correctly
    axes[0, 1].imshow(egg_mask, cmap='gray')
    axes[0, 1].set_title('Digital Cookie Cutter (ROI Mask)')
    
    axes[1, 0].imshow(skeleton, cmap='gray')
    axes[1, 0].set_title(f'Skeletonized (Total Length: {total_length} px)')
    
    img_overlay = img.copy()
    y, x = np.where(branch_points)
    for i in range(len(x)):
        cv2.circle(img_overlay, (x[i], y[i]), 3, (0, 0, 255), -1) 
        
    axes[1, 1].imshow(cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Detected Intersections: {num_branches}')
    
    for ax in axes.ravel():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

    return {
        "vessel_length_pixels": int(total_length),
        "branch_points": int(num_branches)
    }

if __name__ == "__main__":
    # Remember to paste your exact image path here again!
    image_file = r"C:\Users\ASUS\OneDrive\Desktop\Cam_Project\sample_cam.png.jpeg" 
    results = analyze_cam_vessels(image_file)
    print("Final Analysis:", results)