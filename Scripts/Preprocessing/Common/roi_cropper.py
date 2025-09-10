import os
import cv2
import numpy as np

def crop_shapes_with_samicross(input_dir, output_dir,
                               min_outer_area=100,   # Minimum area for an outer shape
                               min_inner_area=50,    # Minimum area for a child (Sami-cross candidate)
                               make_transparent=True):
    """
    Processes all images in the input_dir. For each image, detects red-drawn shapes,
    then checks whether a top-level (outer) red shape contains at least one inner red
    contour (assumed to be the Sami-cross). If yes, it crops the region of the outer shape,
    keeping its interior intact while making the area outside the shape transparent (or white).
    
    :param min_outer_area: Minimum area for an outer red shape to be considered.
    :param min_inner_area: Minimum area for an inner contour to be considered a Sami-cross candidate.
    :param make_transparent: If True, outside the shape becomes transparent (output in PNG).
                             Otherwise, the background is filled white.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not open {image_path}. Skipping.")
            continue

        # Convert image to HSV for robust red color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define red ranges (note: red wraps around the HSV hue circle, so we need two ranges)
        lower_red1 = np.array([0, 70, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Use RETR_TREE to get the full hierarchy (top-level and inner contours)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            print(f"No contours found in {filename}.")
            continue
        hierarchy = hierarchy[0]  # simplify hierarchy structure

        shape_count = 0
        for i, contour in enumerate(contours):
            # We only want top-level contours (outer shapes)
            if hierarchy[i][3] != -1:
                continue

            outer_area = cv2.contourArea(contour)
            if outer_area < min_outer_area:
                continue

            # Check for inner contours (children) using the hierarchy.
            # hierarchy[i][2] gives the first child; then use sibling links.
            has_samicross = False
            child_idx = hierarchy[i][2]
            while child_idx != -1:
                child_area = cv2.contourArea(contours[child_idx])
                if child_area >= min_inner_area:
                    has_samicross = True
                    break
                child_idx = hierarchy[child_idx][0]  # move to next sibling

            if not has_samicross:
                # Skip outer shapes that don't seem to contain a Sami-cross
                continue

            # Crop the outer shape region
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w].copy()

            # Create a mask for the ROI where the outer contour is drawn (shifted to ROI coordinates)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            contour_offset = contour - [x, y]
            cv2.drawContours(roi_mask, [contour_offset], -1, 255, -1)

            if make_transparent:
                # Convert ROI to BGRA so we can add an alpha channel
                roi_bgra = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)
                # Use the roi_mask as the alpha channel: inside shape = opaque (255), outside = transparent (0)
                roi_bgra[:, :, 3] = roi_mask
                cropped_shape = roi_bgra
            else:
                # Otherwise, fill the outside of the shape with white
                white_bg = np.full_like(roi, 255)
                white_bg[roi_mask == 255] = roi[roi_mask == 255]
                cropped_shape = white_bg

            # Save the cropped image; if transparent, use PNG
            base_name, _ = os.path.splitext(filename)
            out_filename = f"{base_name}_cropped_samicross_{shape_count}.png" if make_transparent else f"{base_name}_cropped_samicross_{shape_count}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, cropped_shape)
            print(f"Saved {out_path}")
            shape_count += 1

        if shape_count == 0:
            print(f"No shapes with a Sami-cross detected in {filename}.")

# Example usage:
if __name__ == "__main__":
    input_directory = "D:\\Users\\Majid\\RES_NET\\screenshots"
    output_directory = "D:\\Users\\Majid\\RES_NET\\New_Data\\Tissue_Liver"
    
    # Adjust thresholds as needed for your images.
    crop_shapes_with_samicross(input_directory, output_directory,
                               min_outer_area=100,
                               min_inner_area=50,
                               make_transparent=True)

