import cv2
import streamlit as st
import numpy as np
import time
import tempfile
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Page configuration
st.set_page_config(page_title="Smart Traffic Signal System", layout="wide")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Class IDs for vehicles
motorcycle_id = 3
car_id = 2
bus_id = 5
truck_id = 7


# Weight factors for green time calculation
k_motorcycle = 1.0
k_car = 2.0
k_truck_bus = 3.0 #

# Lane names
lanes = ["Lane 1", "Lane 2", "Lane 3", "Lane 4"]

# Historical data storage
green_time_history = {lane: [] for lane in lanes}

# Function to fetch real-time weather data
def get_weather_conditions():
    api_key = "your_api_key_here"
    city = "your_city_here"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        weather = data["weather"][0]["main"].lower()
        return weather
    except Exception as e:
        st.sidebar.error("⚠️ Unable to fetch weather data.")
        return "clear"

# Function to share real-time traffic light data with navigation apps
def share_traffic_data(lane, green_time):
    api_key = "d1f2fd9d2fmshfc19171c024ea6ap129209jsn76ae5303d592"
    api_url = f"https://maps.googleapis.com/maps/api/directions/json?origin=Berlin&destination=Munich&departure_time=now&traffic_model=best_guess&key={api_key}"


    try:
        response = requests.get(api_url)
        print(response.status_code, response.text)  # Debugging output
        if response.status_code == 200:
            st.sidebar.success("✅ Traffic data retrieved successfully!")
        else:
            st.sidebar.warning(f"⚠️ Failed to fetch traffic data: {response.text}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to traffic API: {e}")

    payload = {
        "lane": lane,
        "green_time": green_time,
        "timestamp": time.time()
    }
    try:
        response = requests.post(api_url, json=payload)
        print(f"API Response: {response.status_code}, {response.text}")  # Debugging line
        if response.status_code == 200:
            st.sidebar.success(f"✅ Traffic data shared successfully for {lane}.")
        else:
            st.sidebar.warning(f"⚠️ Failed to share traffic data for {lane}: {response.text}")
    except Exception as e:
        st.sidebar.error(f"⚠️ Error connecting to navigation API: {e}")




# Function to adjust detection sensitivity based on weather
def adjust_sensitivity(weather):
    if "rain" in weather or "fog" in weather:
        return 0.8  # Decrease sensitivity
    elif "night" in weather or "low light" in weather:
        return 0.9  # Slightly decrease sensitivity
    return 1.0  # Default sensitivity

# Function to estimate distance based on bounding box height
def estimate_distance(bbox_height):
    return 5000 / (bbox_height + 1e-6)

# Function to simulate public transport integration
def update_public_transport(bus_count):
    if bus_count > 0:
        st.sidebar.success(f"🚌 {bus_count} buses detected. Adjusting public transport schedules...")
    else:
        st.sidebar.info("🚏 No buses detected. Public transport schedule remains unchanged.")

# Function to allow users to report traffic issues
def report_issue():
    st.sidebar.markdown("### 🚨 Report an Issue")
    issue_type = st.sidebar.selectbox("Issue Type", ["Malfunctioning Traffic Light", "Accident", "Road Block", "Other"])
    issue_description = st.sidebar.text_area("Describe the issue:")
    if st.sidebar.button("Submit Report"):
        with open("traffic_reports.log", "a") as log_file:
            log_file.write(f"Issue Type: {issue_type}\nDescription: {issue_description}\n---\n")
        st.sidebar.success("✔ Issue reported successfully! Authorities will be notified.")

# Function to predict future green times
def predict_future_green_times():
    st.markdown("### 📈 Future Green Time Prediction")
    future_predictions = {lane: np.mean(history) if history else 10 for lane, history in green_time_history.items()}
    df = pd.DataFrame(list(future_predictions.items()), columns=["Lane", "Predicted Green Time (sec)"])
    st.table(df)
    plt.figure(figsize=(8, 4))
    plt.bar(future_predictions.keys(), future_predictions.values(), color='skyblue')
    plt.xlabel("Lane")
    plt.ylabel("Predicted Green Time (sec)")
    plt.title("Predicted Future Green Times")
    st.pyplot(plt)

# Function to process multiple camera feeds for a lane
def process_multiple_cameras(camera_feeds):
    combined_frame = None
    for feed in camera_feeds:
        cap = cv2.VideoCapture(feed)
        ret, frame = cap.read()
        if ret:
            if combined_frame is None:
                combined_frame = frame
            else:
                combined_frame = cv2.addWeighted(combined_frame, 0.5, frame, 0.5, 0)
        cap.release()
    return combined_frame

# Display issue reporting form
report_issue()

# Display predicted future green times
predict_future_green_times()

# The rest of the original code remains unchanged...




# Function to calculate green time and vehicle counts
def calculate_green_time(frame):
    if frame is None:
        return 10, 0, 0, 0, 0  # Ensure five values are returned

    results = model(frame)
    if not results or not hasattr(results[0], "boxes") or results[0].boxes is None:
        return 10, 0, 0, 0, 0  # Ensure five values are returned

    detected_objects = results[0].boxes.data
    motorcycle_count, car_count, truck_bus_count = 0, 0, 0

    for obj in detected_objects:
        x1, y1, x2, y2, conf, class_id = obj.tolist()
        if class_id == motorcycle_id:
            motorcycle_count += 1
        elif class_id == car_id:
            car_count += 1
        elif class_id in [bus_id, truck_id]:
            truck_bus_count += 1


    green_time = (k_motorcycle * motorcycle_count) + (k_car * car_count) + (k_truck_bus * truck_bus_count)
    green_time = max(5, min(60, green_time))

    return green_time, motorcycle_count, car_count, truck_bus_count
  # Ensure five values are returned


  # Now returning 5 values!


# Function to create a heatmap overlay
def create_heatmap(frame, detected_objects):
    heatmap = np.zeros_like(frame[:, :, 0]).astype(np.float32)  # Create a blank heatmap canvas

    for obj in detected_objects:
        x1, y1, x2, y2, conf, class_id = obj.tolist()
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Add heat to the heatmap at the center of the bounding box
        heatmap[center_y, center_x] += 1

    # Apply Gaussian blur to smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)  # Normalize heatmap values
    heatmap = np.uint8(heatmap)

    # Convert heatmap to a color map (e.g., Jet)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)  # Overlay heatmap on the frame

    return heatmap_colored

# Sidebar input for video stream URL
st.sidebar.markdown("### Enter IP Webcam Stream URL")
video_url = st.sidebar.text_input("Ngrok/Local URL:", "")

if video_url:
    if not video_url.startswith("http://") and not video_url.startswith("https://"):
        video_url = "http://" + video_url

    st.sidebar.write(f"Attempting to connect to: {video_url}")
    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        st.error("⚠ Unable to open video stream. Please check the URL and ensure the camera is accessible.")
    else:
        st.success("✔ Successfully connected to the video stream!")
        stframe = st.empty()
        start_time = time.time()

        # Initialize lane statistics for live stream
        lane_stats = {lane: {"total_vehicles": 0, "total_green_time": 0, "frames_processed": 0} for lane in lanes}

        # Lane switching logic for live stream
        active_lane = 0
        lane_change_time = 60  # Maximum time for each lane
        no_vehicle_timeout = 5  # Time to switch lanes if no vehicles are detected
        last_detection_time = time.time()
        lane_start_time = time.time()

        # Statistics display section
        st.markdown("---")
        st.markdown("### 📊 Lane Statistics (Live Stream)")

        while True:
            elapsed_time = time.time() - lane_start_time
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠ No frames received from the stream.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            detected_objects = results[0].boxes.data if results and hasattr(results[0], "boxes") else []

            # Create heatmap overlay
            heatmap_frame = create_heatmap(frame_rgb, detected_objects)

            green_time, motorcycle_count, car_count, truck_bus_count, = calculate_green_time(frame_rgb)





            # Update lane statistics for live stream
            lane_stats[lanes[active_lane]]["total_vehicles"] += motorcycle_count + car_count + truck_bus_count
            lane_stats[lanes[active_lane]]["total_green_time"] += green_time
            lane_stats[lanes[active_lane]]["frames_processed"] += 1

            # Display the live stream with heatmap overlay
            text = f"Lane {lanes[active_lane]} | Motorcycles: {motorcycle_count} | Cars: {car_count} | Trucks/Buses: {truck_bus_count} | Green Time: {green_time:.2f} sec"
            cv2.putText(heatmap_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            stframe.image(heatmap_frame, channels="RGB", use_column_width=True)

            # Check if no vehicles are detected for 5 seconds
            if motorcycle_count == 0 and car_count == 0 and truck_bus_count == 0:
                if time.time() - last_detection_time >= no_vehicle_timeout:
                    active_lane = (active_lane + 1) % 4  # Switch to the next lane
                    lane_start_time = time.time()  # Reset the lane timer
                    last_detection_time = time.time()  # Reset the detection timer
                    st.warning(f"⚠ No vehicles detected for {no_vehicle_timeout} seconds. Switching to {lanes[active_lane]}.")
            else:
                last_detection_time = time.time()  # Reset the detection timer if vehicles are detected

            # Check if 60 seconds have elapsed for the current lane
            if elapsed_time >= lane_change_time:
                active_lane = (active_lane + 1) % 4  # Switch to the next lane
                lane_start_time = time.time()  # Reset the lane timer
                st.warning(f"⚠ Maximum time ({lane_change_time} seconds) elapsed. Switching to {lanes[active_lane]}.")

            # Update and display lane statistics dynamically
            st.markdown("---")
            st.markdown("### 📊 Lane Statistics (Live Stream)")
            for lane, stats in lane_stats.items():
                avg_green_time = stats["total_green_time"] / stats["frames_processed"] if stats["frames_processed"] > 0 else 0
                st.markdown(f"#### {lane}")
                st.markdown(f"- **Total Vehicles Detected**: {stats['total_vehicles']}")
                st.markdown(f"- **Total Green Time Allocated**: {stats['total_green_time']:.2f} sec")
                st.markdown(f"- **Average Green Time per Frame**: {avg_green_time:.2f} sec")
                st.markdown(f"- **Frames Processed**: {stats['frames_processed']}")

            time.sleep(0.03)
else:
    st.warning("⚠ Please enter a valid video stream URL.")

weather = get_weather_conditions()
st.sidebar.markdown(f"**🌤 Current Weather:** {weather.capitalize()}")
if st.sidebar.button("🔄 Update Navigation Apps"):
    for lane in lanes:
        avg_green_time = np.mean(green_time_history[lane]) if green_time_history[lane] else 10
        share_traffic_data(lane, avg_green_time)



# File uploader for videos
st.sidebar.markdown("### 📂 Upload Four Videos")
uploaded_files = st.file_uploader("Select 4 Videos", type=["mp4"], accept_multiple_files=True)

if len(uploaded_files) == 4:
    temp_videos = []
    for file in uploaded_files:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(file.read())
        temp_videos.append(temp_video.name)

    st.sidebar.success("✔ Four videos uploaded successfully!")
    stframes = [st.empty() for _ in range(4)]
    progress_bar = st.sidebar.progress(0)
    caps = [cv2.VideoCapture(video) for video in temp_videos]

    start_time = time.time()
    lane_change_time = 60
    active_lane = 0

    # Initialize lane statistics
    lane_stats = {lane: {"total_vehicles": 0, "total_green_time": 0, "frames_processed": 0} for lane in lanes}

    # Statistics display section
    st.markdown("---")
    st.markdown("### 📊 Lane Statistics (Uploaded Videos)")

    while any(cap.isOpened() for cap in caps):
        elapsed_time = time.time() - start_time
        cap = caps[active_lane]

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detected_objects = results[0].boxes.data if results and hasattr(results[0], "boxes") else []

        # Create heatmap overlay
        heatmap_frame = create_heatmap(frame_rgb, detected_objects)

        # Calculate green time and vehicle counts
        green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame_rgb)

        # Update lane statistics
        lane_stats[lanes[active_lane]]["total_vehicles"] += motorcycle_count + car_count + truck_bus_count
        lane_stats[lanes[active_lane]]["total_green_time"] += green_time
        lane_stats[lanes[active_lane]]["frames_processed"] += 1

        # Display the video with heatmap overlay
        text = f"Lane {lanes[active_lane]} | Motorcycles: {motorcycle_count} | Cars: {car_count} | Trucks/Buses: {truck_bus_count} | Green Time: {green_time:.2f} sec"
        cv2.putText(heatmap_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        stframes[active_lane].image(heatmap_frame, channels="RGB", use_column_width=True)

        progress_bar.progress(min(int(elapsed_time / lane_change_time * 100), 100))

        if elapsed_time >= lane_change_time or (
                motorcycle_count == 0 and car_count == 0 and truck_bus_count == 0 and elapsed_time >= 5):
            active_lane = (active_lane + 1) % 4
            start_time = time.time()

        # Update and display lane statistics dynamically
        st.markdown("---")
        st.markdown("### 📊 Lane Statistics (Uploaded Videos)")
        for lane, stats in lane_stats.items():
            avg_green_time = stats["total_green_time"] / stats["frames_processed"] if stats["frames_processed"] > 0 else 0
            st.markdown(f"#### {lane}")
            st.markdown(f"- **Total Vehicles Detected**: {stats['total_vehicles']}")
            st.markdown(f"- **Total Green Time Allocated**: {stats['total_green_time']:.2f} sec")
            st.markdown(f"- **Average Green Time per Frame**: {avg_green_time:.2f} sec")
            st.markdown(f"- **Frames Processed**: {stats['frames_processed']}")

        time.sleep(0.03)
else:
    st.warning("⚠ Please upload exactly 4 videos to proceed.")

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
                cap.release()
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if elapsed_time < lane_change_time:
                green_time, motorcycle_count, car_count, truck_bus_count = calculate_green_time(frame_rgb)


                text = f"🚦 {lanes[current_lane_index]} | 🚲 Motorcycles: {motorcycle_count} | 🚗 Cars: {car_count} | 🚌 Trucks/Buses: {truck_bus_count} | ⏳ Green Time: {green_time:.2f} sec"
                cv2.putText(frame, text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                text = f"🛑 {lanes[current_lane_index]} stopped. Moving to next lane."
                cv2.putText(frame, text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            stframe.image(frame, channels="BGR", use_column_width=False, width=800)

        cap.release()
        progress_bar.empty()

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F3E9D2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply the background color
set_background_color()

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



