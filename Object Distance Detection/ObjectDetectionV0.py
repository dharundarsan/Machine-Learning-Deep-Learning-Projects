import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 pretrained on COCO

# Function to calculate distance from stereo images
def find_dist(pt1, pt2, f1, f2, baseline, f, alpha):
    h1, w1, ch1 = f1.shape
    h2, w2, ch2 = f2.shape
    
    # Ensure both frames have the same width
    if w1 != w2:
        raise ValueError("Frames have different widths")

    x0 = baseline / 2
    y0 = 0

    f_pixel = (w1 * 0.5) / (np.tan(alpha * 0.5 * np.pi / 180))

    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]

    disparity = x1 - x2
    dist_z = ((baseline * f_pixel) / disparity)

    dist_x = (x1 - w1 // 2) * dist_z / f_pixel + x0
    dist_y = (y1 - h1 // 2) * dist_z / f_pixel - y0

    return dist_x, dist_y, dist_z

# Function to detect cars and calculate distance
def detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha):
    # Resize frames to match dimensions
    h_left, w_left, _ = left_frame.shape
    h_right, w_right, _ = right_frame.shape

    if (h_left != h_right) or (w_left != w_right):
        right_frame = cv2.resize(right_frame, (w_left, h_left))

    results_left = model(left_frame)
    results_right = model(right_frame)

    car_left = results_left[0].boxes.xywh[0][:2].cpu().numpy()  # Left image car center
    car_right = results_right[0].boxes.xywh[0][:2].cpu().numpy()  # Right image car center

    dist_x, dist_y, dist_z = find_dist(car_left, car_right, left_frame, right_frame, baseline, focal_length_mm, alpha)

    return dist_x, dist_y, dist_z

# Load stereo images (replace with actual images)
left_frame = cv2.imread('D:\\mini project\\left1.jpg')
right_frame = cv2.imread('D:\\mini project\\right1.jpg')

# Check if images are loaded properly
if left_frame is None or right_frame is None:
    print("Error: Unable to load images.")
else:
    print("Images loaded successfully.")

baseline = 0.064  # Baseline distance between stereo cameras in meters
focal_length_mm = 3.6  # Focal length of the camera in mm
alpha = 70  # Field of view of the camera in degrees

# Calculate the distance of the detected car
dist_x, dist_y, dist_z = detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha)

print(f"Detected car distance: X = {dist_x:.2f} meters, Y = {dist_y:.2f} meters, Z = {dist_z:.2f} meters")
