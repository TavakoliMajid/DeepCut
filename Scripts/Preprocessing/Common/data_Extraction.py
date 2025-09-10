#video_path = "D:\\Users\\Majid\\OD\\Old_Method\\Videos\\Video_1.avi"
#roi_x, roi_y, roi_width, roi_height = 180, 230, 400, 170
import cv2
import numpy as np
import os

def detect_red(frame):
    """
    Detect red color in the given frame (or ROI).
    Draws a black rectangle over the detected red areas and returns True if red is detected.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define red color range in HSV
    lower_red1 = np.array([0, 120, 70])   # First range of red
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70]) # Second range of red
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours of the red areas
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw black rectangles over detected red areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)  # Black filled rectangle

    # Check if red is detected
    red_pixels = cv2.countNonZero(red_mask)  # Count non-zero pixels
    threshold = 500  # Minimum pixels to consider red detected
    return red_pixels > threshold

# Create a folder for screenshots
os.makedirs("screenshots", exist_ok=True)

# Initialize video capture
video_path = "D:\\Users\\Majid\\OD\\Old_Method\\Videos\\New-Videos\\Video_11.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Grab the first frame to define ROIs interactively
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video file.")
    cap.release()
    exit()

# Select the first ROI (for screenshots)
print("Select the ROI for screenshots (click and drag), then press ENTER or SPACE.")
screenshot_roi = cv2.selectROI("Select Screenshot ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Screenshot ROI")

# Extract coordinates for the screenshot ROI
screenshot_roi_x, screenshot_roi_y, screenshot_roi_width, screenshot_roi_height = map(int, screenshot_roi)
print(f"Screenshot ROI selected: x={screenshot_roi_x}, y={screenshot_roi_y}, width={screenshot_roi_width}, height={screenshot_roi_height}")

# Select the second ROI (for red detection)
print("Select the ROI for red detection (click and drag), then press ENTER or SPACE.")
detection_roi = cv2.selectROI("Select Red Detection ROI", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Red Detection ROI")

# Extract coordinates for the detection ROI
detection_roi_x, detection_roi_y, detection_roi_width, detection_roi_height = map(int, detection_roi)
print(f"Red Detection ROI selected: x={detection_roi_x}, y={detection_roi_y}, width={detection_roi_width}, height={detection_roi_height}")

red_detected = False  # Track if red was detected in the last frame
frame_count = 0       # For screenshot naming

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    # Extract the red detection ROI from the frame
    detection_roi_frame = frame[
        detection_roi_y:detection_roi_y + detection_roi_height, 
        detection_roi_x:detection_roi_x + detection_roi_width
    ]
    red_in_frame = detect_red(detection_roi_frame)  # Perform red detection in the selected ROI
    
    # Display the result of red detection on the screen
    if red_in_frame:
        red_detected = True  # Red detected
        cv2.putText(frame, "Red Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if red_detected:  # Red was detected in the previous frame
            red_detected = False  # Red no longer detected, take screenshot
            
            # Extract the screenshot ROI from the frame
            screenshot_roi_frame = frame[
                screenshot_roi_y:screenshot_roi_y + screenshot_roi_height, 
                screenshot_roi_x:screenshot_roi_x + screenshot_roi_width
            ]
            
            # Save the screenshot
            frame_count += 1
            screenshot_filename = f"screenshots/screenshot22_{frame_count}.jpg"
            cv2.imwrite(screenshot_filename, screenshot_roi_frame)
            print(f"Screenshot saved: {screenshot_filename}")
        
        cv2.putText(frame, "No Red Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw both ROIs on the frame for visualization
    # Screenshot ROI (Blue)
    cv2.rectangle(frame, 
                  (screenshot_roi_x, screenshot_roi_y), 
                  (screenshot_roi_x + screenshot_roi_width, screenshot_roi_y + screenshot_roi_height), 
                  (255, 0, 0), 2)  # Blue for screenshot ROI
    
    # Detection ROI (Green)
    cv2.rectangle(frame, 
                  (detection_roi_x, detection_roi_y), 
                  (detection_roi_x + detection_roi_width, detection_roi_y + detection_roi_height), 
                  (0, 255, 0), 2)  # Green for detection ROI
    
    # Show the frame with both ROIs and detection results
    cv2.imshow("Video", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Video processing stopped by user.")
        break

cap.release()
cv2.destroyAllWindows()
