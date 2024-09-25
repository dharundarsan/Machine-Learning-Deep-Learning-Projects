from ultralytics import YOLO
import cv2
import numpy as np

# Distance calculation function
def find_dist(pt1, pt2, f1, f2, baseline, f, alpha):
    h1, w1, ch1 = f1.shape
    h2, w2, ch2 = f2.shape
    x0 = baseline / 2
    y0 = 0  # We consider both the cameras at the same height
    
    if w1 == w2:
        # Converting focal length from mm to pixels
        f_pixel = (w1 * 0.5) / (np.tan(alpha * 0.5 * np.pi / 180))
    else:
        print("Frames have different widths")
        return None

    x1, x2 = pt1[0], pt2[0]
    y1, y2 = pt1[1], pt2[1]
    
    disparity = x1 - x2  # Difference in x-coordinates
    if disparity == 0:
        print("Disparity is zero, cannot compute distance")
        return None
    
    dist_z = (baseline * f_pixel) / disparity
    dist_x = (x1 - (w1 // 2)) * dist_z / f_pixel
    dist_x += x0  # Measuring from center of the stereo system
    dist_y = y1 * dist_z / f_pixel
    dist_y -= y0  # Adjust for the height (if required)
    
    return dist_x, dist_y, dist_z

# Object detection function with stereo image distance calculation
def detect_objects_in_stereo_images(left_image_path, right_image_path, baseline, focal_length, fov):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")  # Specify your model if different

    # Load left and right images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Perform object detection on the left image
    left_results = model(left_image_path)
    print(f"Results for Left Image ({left_image_path}):")
    left_boxes = []
    for result in left_results:
        print(f"Detected {len(result.boxes)} objects")
        for box in result.boxes:
            cls = box.cls.item()  # Convert to a scalar
            conf = box.conf.item()  # Convert to a scalar
            xyxy = box.xyxy.cpu().numpy().flatten()  # Flatten the array to handle single coordinates
            left_boxes.append(xyxy)
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

    # Perform object detection on the right image
    right_results = model(right_image_path)
    print(f"\nResults for Right Image ({right_image_path}):")
    right_boxes = []
    for result in right_results:
        print(f"Detected {len(result.boxes)} objects")
        for box in result.boxes:
            cls = box.cls.item()  # Convert to a scalar
            conf = box.conf.item()  # Convert to a scalar
            xyxy = box.xyxy.cpu().numpy().flatten()  # Flatten the array to handle single coordinates
            right_boxes.append(xyxy)
            print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

    # Find matching objects between left and right images
    if not left_boxes or not right_boxes:
        print("No matching objects found in both images.")
        return

    for i, left_box in enumerate(left_boxes):
        for j, right_box in enumerate(right_boxes):
            # Match objects based on proximity (e.g., same class and coordinate similarity)
            if np.abs(left_box[1] - right_box[1]) < 50 and np.abs(left_box[3] - right_box[3]) < 50:
                left_center = [(left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3]) / 2]
                right_center = [(right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3]) / 2]

                # Calculate distance
                dist_x, dist_y, dist_z = find_dist(left_center, right_center, left_image, right_image, baseline, focal_length, fov)
                print(f"\nMatching object found between images {i + 1} and {j + 1}:")
                print(f"Left Image Center: {left_center}, Right Image Center: {right_center}")
                print(f"Distance: X={dist_x:.2f}m, Y={dist_y:.2f}m, Z={dist_z:.2f}m")

    # Optionally, visualize the results for both images
    left_results[0].show()  # Show the left image with bounding boxes
    right_results[0].show()  # Show the right image with bounding boxes

# Example usage
left_image_path = "D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left3.png"
right_image_path = "D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\right3.png"

# Stereo camera parameters
baseline = 0.064  # Distance between the two cameras (in meters)
focal_length = 3.6  # Focal length of the camera lens (in mm)
fov = 90  # Field of view (in degrees)

detect_objects_in_stereo_images(left_image_path, right_image_path, baseline, focal_length, fov)
