import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Vehicle Classes (COCO IDs)
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

# Time weights (adjustable based on experiments)
k_motorcycle = 1.0  # Seconds per motorcycle
k_car = 2.0  # Seconds per car
k_truck_bus = 3.0  # Seconds per truck/bus

# Function to calculate green light duration for each frame
def calculate_green_time(frame):
    if frame is None:
        return 10, 0, 0, 0  # Default values if frame is not loaded

    results = model(frame)

    if len(results) == 0 or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 10, 0, 0, 0  # Minimum green time if no detections

    detected_objects = results[0].boxes.data

    motorcycle_count = 0
    car_count = 0
    truck_bus_count = 0

    for obj in detected_objects:
        class_id = int(obj[5])
        if class_id == motorcycle_id:
            motorcycle_count += 1
        elif class_id == car_id:
            car_count += 1
        elif class_id in [bus_id, truck_id]:
            truck_bus_count += 1

    green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)
    green_time = max(10, min(60, green_time))  # Constrain to 10-60 seconds

    return green_time, motorcycle_count, car_count, truck_bus_count

# Streamlit UI
st.title("🚦 Smart Traffic Signal System with Image & Video Processing")

# File uploader (supports images & videos)
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "jpeg", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # **If the uploaded file is an IMAGE**
    if "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Object detection
        green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(img_rgb)

        # Overlay text
        text = f"Motorcycles: {motorcycle_count} | Cars: {car_count} | Trucks/Buses: {truck_bus_count} | Green Time: {green_time:.2f} sec"
        cv2.putText(img_rgb, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Display the image
        st.image(img_rgb, caption="Processed Image", use_column_width=True)

    # **If the uploaded file is a VIDEO**
    elif "video" in file_type:
        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())

        # Open video using OpenCV
        cap = cv2.VideoCapture(temp_video.name)

        stframe = st.empty()  # Placeholder for displaying video
        start_time = time.time()  # Start time tracking
        detection_active = True

        while cap.isOpened():
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            ret, frame = cap.read()
            if not ret:
                break

            if elapsed_time >= 60:
                detection_active = False  # Stop detection after 60 seconds

            # Convert frame color for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if detection_active:
                # Object detection and green time calculation
                green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame_rgb)

                # Overlay information on frame
                text = f"Motorcycles: {motorcycle_count} | Cars: {car_count} | Trucks/Buses: {truck_bus_count} | Green Time: {green_time:.2f} sec"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                text = "Allotted time is over! No more vehicles detected."
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()



