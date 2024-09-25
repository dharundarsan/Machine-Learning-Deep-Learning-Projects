from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Specify your model if different

# Perform object detection on an image
results = model("D:\\mini project\\Machine-Learning-Deep-Learning-Projects\\Object Distance Detection\\left5.jpg")

# Iterate through results and print detection details
for result in results:
    print(f"Detected {len(result.boxes)} objects")
    for box in result.boxes:
        # Convert tensor values to NumPy arrays or Python scalars
        cls = box.cls.item()  # Convert to a scalar
        conf = box.conf.item()  # Convert to a scalar
        xyxy = box.xyxy.cpu().numpy()  # Convert to a NumPy array

        # Print detection details
        print(f"Class: {cls}, Confidence: {conf:.2f}, Coordinates: {xyxy}")

# Alternatively, you can visualize the results
results[0].show()  # Show the image with bounding boxes
