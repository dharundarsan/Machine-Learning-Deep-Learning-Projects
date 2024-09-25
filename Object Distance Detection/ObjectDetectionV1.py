import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('yolov8n.pt')  # YOLOv8 pretrained on COCO

# Function to calculate distance from stereo images
def find_dist(pt1, pt2, baseline, focal_length_mm, alpha):
    if pt1.size != 2 or pt2.size != 2:
        raise ValueError("Points must be of shape (2,)")

    x1, y1 = pt1
    x2, y2 = pt2

    disparity = x1 - x2
    if disparity == 0:
        raise ValueError("Disparity cannot be zero for distance calculation.")

    f_pixel = (640 * 0.5) / (np.tan(alpha * 0.5 * np.pi / 180))  # Assuming width is 640

    dist_z = (baseline * f_pixel) / disparity
    dist_x = (x1 - 640 // 2) * dist_z / f_pixel
    dist_y = (y1 - 480 // 2) * dist_z / f_pixel  # Assuming height is 480

    return dist_x, dist_y, dist_z

# Function to compare detected objects and find matches
def compare_detections(det_left, det_right):
    if len(det_left) == 0 or len(det_right) == 0:
        return None

    for left_box in det_left:
        for right_box in det_right:
            if (np.abs(left_box[0] - right_box[0]) < 30 and  # X-coordinate
                np.abs(left_box[1] - right_box[1]) < 30):  # Y-coordinate
                return left_box[:2], right_box[:2]  # Return the centers

    return None

# Function to detect cars and calculate distance
def detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha):
    left_frame_resized = cv2.resize(left_frame, (640, 480))
    right_frame_resized = cv2.resize(right_frame, (640, 480))

    results_left = model(left_frame_resized)
    results_right = model(right_frame_resized)

    # Get the detected bounding boxes and class IDs
    det_left = results_left[0].boxes.numpy()  # Use numpy() instead of cpu()
    det_right = results_right[0].boxes.numpy()  # Use numpy() instead of cpu()

    # Filter detections to only include cars (class ID 2)
    det_left_cars = [box.xywh for box in det_left if int(box.cls) == 2]  # Filter cars
    det_right_cars = [box.xywh for box in det_right if int(box.cls) == 2]  # Filter cars

    # Ensure we are handling the car detections correctly
    det_left_cars = np.array(det_left_cars)
    det_right_cars = np.array(det_right_cars)

    matched_cars = compare_detections(det_left_cars, det_right_cars)

    if matched_cars is None:
        print("No matching cars detected in both images.")
        return None, None, None, results_left, results_right

    car_left, car_right = matched_cars
    dist_x, dist_y, dist_z = find_dist(car_left, car_right, baseline, focal_length_mm, alpha)

    return dist_x, dist_y, dist_z, results_left, results_right

# Load stereo images
left_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left2.jpg')
right_frame = cv2.imread('D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\right1.jpg')

# Check if images are loaded properly
if left_frame is None or right_frame is None:
    print("Error: Unable to load images.")
else:
    print("Images loaded successfully.")

baseline = 0.064  # Baseline distance between stereo cameras in meters
focal_length_mm = 3.6  # Focal length of the camera in mm
alpha = 70  # Field of view of the camera in degrees

# Calculate the distance of the detected car
dist_x, dist_y, dist_z, results_left, results_right = detect_and_calculate_distance(left_frame, right_frame, baseline, focal_length_mm, alpha)

if dist_x is not None and dist_y is not None and dist_z is not None:
    print(f"Detected car distance: X = {dist_x:.2f} meters, Y = {dist_y:.2f} meters, Z = {dist_z:.2f} meters")
else:
    print("Distance calculation failed.")

# Visualize detected bounding boxes
for box in results_left[0].boxes.numpy():  # Updated to use numpy()
    if int(box.cls) == 2:  # Check if the detected object is a car
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

for box in results_right[0].boxes.numpy():  # Updated to use numpy()
    if int(box.cls) == 2:  # Check if the detected object is a car
        x1, y1, x2, y2 = map(int, box.xyxy)
        cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show the frames with detected bounding boxes
cv2.imshow("Left Frame", left_frame)
cv2.imshow("Right Frame", right_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
