import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from ultralytics import YOLO

st.set_page_config(page_title="Smart Traffic Signal System", layout="wide")

# Initialize session state for back to top
if "back_to_top" not in st.session_state:
    st.session_state.back_to_top = False

model = YOLO("yolov8n.pt")

motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7

k_motorcycle = 1.0
k_car = 2.0
k_truck_bus = 3.0

lanes = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]
current_lane_index = 0

def estimate_distance(bbox_height):
    return 5000 / (bbox_height + 1e-6)

def calculate_green_time(frame):
    if frame is None:
        return 10, 0, 0, 0

    results = model(frame)

    if len(results) == 0 or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 10, 0, 0, 0

    detected_objects = results[0].boxes.data
    motorcycle_count, car_count, truck_bus_count = 0, 0, 0

    for obj in detected_objects:
        x1, y1, x2, y2, conf, class_id = obj.tolist()
        bbox_height = y2 - y1
        distance = estimate_distance(bbox_height)

        if distance > 80:
            continue

        class_id = int(class_id)
        if class_id == motorcycle_id:
            motorcycle_count += 1
        elif class_id == car_id:
            car_count += 1
        elif class_id in [bus_id, truck_id]:
            truck_bus_count += 1

    green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)
    green_time = max(10, min(60, green_time))

    return green_time, motorcycle_count, car_count, truck_bus_count

st.title("🚦 Smart Traffic Signal System with Sequential Lane Switching")
st.markdown("---")

# Sidebar with content
uploaded_file = st.file_uploader("📂 Upload an Image or Video", type=["jpg", "png", "jpeg", "mp4"],
                                 help="Supports JPG, PNG, and MP4 formats")

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(img_rgb)

        text = f"🚲 Motorcycles: {motorcycle_count} | 🚗 Cars: {car_count} | 🚌 Trucks/Buses: {truck_bus_count} | ⏳ Green Time: {green_time:.2f} sec"
        cv2.putText(img_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        st.image(img_rgb, caption="🖼 Processed Image", use_column_width=True)

    elif "video" in file_type:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()
        start_time = time.time()
        detection_active = True
        lane_change_time = 60
        current_lane_index = 0

        st.sidebar.markdown("### 🎥 Video Processing")
        progress_bar = st.sidebar.progress(0)

        while cap.isOpened():
            elapsed_time = time.time() - start_time
            ret, frame = cap.read()
            if not ret:
                break

            progress_bar.progress(min(int(elapsed_time / lane_change_time * 100), 100))

            if elapsed_time >= lane_change_time:
                current_lane_index = (current_lane_index + 1) % len(lanes)
                start_time = time.time()
                st.sidebar.write(f"🔄 Switching to {lanes[current_lane_index]}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if elapsed_time < lane_change_time:
                green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame_rgb)
                text = f"🚦 {lanes[current_lane_index]} | 🚲 Motorcycles: {motorcycle_count} | 🚗 Cars: {car_count} | 🚌 Trucks/Buses: {truck_bus_count} | ⏳ Green Time: {green_time:.2f} sec"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                text = f"🛑 {lanes[current_lane_index]} stopped. Moving to next lane."
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=False, width=800)

        cap.release()
        progress_bar.empty()

# Sidebar Content
st.sidebar.markdown("💡 **How it Works**")
st.sidebar.write("1. Upload an image or video.")
st.sidebar.write("2. The system detects vehicles within 80m.")
st.sidebar.write("3. Calculates optimal green light duration.")
st.sidebar.write("4. Switches lanes every 60 seconds.")
st.sidebar.write("5. Displays processed output with insights.")

st.sidebar.markdown("📚 **More Databases**")
st.sidebar.write("Explore more traffic-related datasets on Kaggle:")
st.sidebar.write("[Kaggle Traffic Datasets](https://www.kaggle.com/datasets?search=traffic)")

st.sidebar.markdown("📞 **Contact**")
st.sidebar.write("[LinkedIn Profile](https://www.linkedin.com/in/anand-kumar-91461a19a)")

st.sidebar.markdown("🛠 **Help & Support**")
st.sidebar.write("- [GitHub Issues](https://github.com/Anandkumae/Adaptivetraffic1-app/issues)")
st.sidebar.write("- [LinkedIn](https://www.linkedin.com/in/anand-kumar-91461a19a)")
st.sidebar.write("- Local Developer Contact: anandkumar06091561@gmail.com")

st.sidebar.markdown("---")
st.sidebar.markdown("English(India)")
st.sidebar.markdown("---")

# Footer Content
st.sidebar.markdown("© 2025 **ats2025**. All rights reserved.")

# "Back to Top" button at the very end of the sidebar
back_to_top = st.sidebar.button("🔝 Back to Top")

# Using JavaScript to Scroll Sidebar
if back_to_top:
    st.markdown(
        """
        <script>
            const sidebar = document.querySelector('div[data-testid="stSidebar"]');
            sidebar.scrollTop = 0;
        </script>
        """,
        unsafe_allow_html=True,
    )



