import cv2
import numpy as np
import datetime
import os
import pandas as pd
import time
from time import sleep
from threading import Thread, Event
from flask import Flask, render_template, jsonify, send_file, request
import plotly
from plotly.utils import PlotlyJSONEncoder
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
from picamera import PiCamera
from picamera import mmal, mmalobj, exc
from picamera.mmalobj import to_rational
from picamera.array import PiRGBArray
import subprocess
import json
import serial
from threading import Lock
from pykalman import KalmanFilter
import random



serial_lock = Lock()
# Global variables to be set by /run_program
program_trigger = False
selected_program = None
selected_temperature = None
selected_t1 = None
selected_process_time = None
program_result = None
start_time = None
program_status = None
elapsed_time = 0
last_five_hues_n = []  # Store the last 5 hues for Tube N

COMPARATOR = 1.4
ORANGE_OFFSET = 20
VALID_PIXEL_THRESHOLD = 80  # Minimum number of valid pixels required
# Directory to save images
image_dir = '/home/lamp/testtubenew/Test_Image'

# Control events for sample collection process
pause_event = Event()
pause_event.set()
stop_event = Event()

ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# Create directory for saving images if it doesn't exist
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Created directory {image_dir}")

# Create DataFrame to store Hue values over time
hue_columns = ['Timestamp'] + [f'Tube_{i}_Hue' for i in range(1, 9)]
df_hue = pd.DataFrame(columns=hue_columns)

# Create DataFrame to store Program results
program_columns = ['Timestamp','Program', 'Tube', 'C Value', 'Result of Tube']
df_program = pd.DataFrame(columns=program_columns)

# Check and create CSV file for hue values if it doesn't exist
hue_csv_file = '/home/lamp/testtubenew/test_tube_hue_values.csv'
if not os.path.exists(hue_csv_file) or os.stat(hue_csv_file).st_size == 0:
    df_hue.to_csv(hue_csv_file, index=False)
    print(f"Created Hue CSV file {hue_csv_file}")
else:
    df_hue = pd.read_csv(hue_csv_file)
    print(f"Loaded Hue CSV file {hue_csv_file}")

# Check and create CSV file for program results if it doesn't exist
program_csv_file = '/home/lamp/testtubenew/program_results.csv'
if not os.path.exists(program_csv_file) or os.stat(program_csv_file).st_size == 0:
    df_program.to_csv(program_csv_file, index=False)
    print(f"Created Program CSV file {program_csv_file}")
else:
    df_program = pd.read_csv(program_csv_file)
    print(f"Loaded Program CSV file {program_csv_file}")


latest_image_path = None
capture_interval = 1

#Regions for each test tube in the image
regions = {
    "tube_1": (30, 48, 60, 80),
    "tube_2": (92, 48, 122, 80),
    "tube_3": (158, 48, 188, 80),
    "tube_4": (227, 48, 257, 80),
    "tube_5": (296, 48, 326, 80),
    "tube_6": (365, 48, 395, 80),
    "tube_7": (436, 48, 466, 80),
    "tube_8": (503, 48, 533, 80)
}
CROP_Y1 = 345
CROP_Y2 = 455
CROP_X1 = 320
CROP_X2 = 870




# Initialize the Kalman filter for 8 test tubes
# Adjust the Kalman filter parameters
kf = [KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                   transition_matrices=1, observation_matrices=1, 
                   initial_state_covariance=1, 
                   transition_covariance=1e-2,  # Increase this slightly for faster adaptation
                   observation_covariance=5e-2)  # Slightly lower observation covariance to rely more on observed data
      for _ in range(8)]

state_means = [np.array([0]) for _ in range(8)]
state_covariances = [np.array([[1]]) for _ in range(8)]




# Global variable to store the history of hue values (for 8 tubes)
hue_history = [[] for _ in range(8)]  # Store the last 5 hue values for each tube

def update_hue_and_calculate_average(hue_value):
    global hue_history

    filtered_hues = []

    for i in range(8):
        if hue_value[i] is not None:
            # Add the current hue value to the history list for this tube
            hue_history[i].append(hue_value[i])

            # Only keep the last 5 values in the history
            if len(hue_history[i]) > 5:
                hue_history[i].pop(0)

            # Filter out None values and calculate the average of the remaining values
            valid_hues = [h for h in hue_history[i] if h is not None]
            if valid_hues:
                average_hue = sum(valid_hues) / len(valid_hues)
            else:
                average_hue = None
        else:
            # If current hue is None, we don't update the history, just use the previous values
            valid_hues = [h for h in hue_history[i] if h is not None]
            if valid_hues:
                average_hue = sum(valid_hues) / len(valid_hues)
            else:
                average_hue = None

        filtered_hues.append(average_hue)

    return filtered_hues




def detect_test_tube(image):
    results = {}
    
    # Define threshold for brightness (V channel) to filter out bright spots
    BRIGHTNESS_THRESHOLD = 240  # Adjust this value based on the actual noise level in your images

    hue_values = []  # List to store hue values for all tubes

    for i, (tube, (x1, y1, x2, y2)) in enumerate(regions.items()):
        sub_image = image[y1:y2, x1:x2]
        blur_sub_image = cv2.GaussianBlur(sub_image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blur_sub_image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]  # Hue channel
        value_channel = hsv_image[:, :, 2]  # V (brightness) channel

        # Filter out pixels that have high brightness values (likely noise)
        mask_bright_pixels = value_channel < BRIGHTNESS_THRESHOLD
        hue_channel_filtered = np.ma.masked_array(hue_channel, mask=~mask_bright_pixels)

        # Define the bounds for red-yellow hues
        lower_bound_1 = 280  # Directly use degrees, not scaled down
        upper_bound_1 = 360
        lower_bound_2 = 0
        upper_bound_2 = 90

        # Convert hue_channel to the correct scale (0-360)
        hue_channel_filtered = hue_channel_filtered.astype(float) * 2

        # Create mask to include only hues within the red-yellow range (280°-360° and 0°-90°)
        mask_hue = ((hue_channel_filtered >= lower_bound_1) | (hue_channel_filtered <= upper_bound_2)).astype(np.uint8)

        # Apply morphological opening to remove small isolated noise
        kernel = np.ones((5, 5), np.uint8)  # Define a 5x5 kernel for morphological operations
        mask_hue = cv2.morphologyEx(mask_hue, cv2.MORPH_OPEN, kernel)

        # Mask the hue values outside the desired range and combine with brightness mask
        final_mask = np.logical_and(mask_hue, mask_bright_pixels)
        masked_hue = np.ma.masked_array(hue_channel_filtered, mask=~final_mask)

        # Filter out noise: Only proceed if there are enough valid pixels
        valid_pixel_count = np.ma.count(masked_hue)
        if valid_pixel_count < VALID_PIXEL_THRESHOLD:
            mean_scaled_hue = None
        else:
            scaled_hue = np.zeros_like(masked_hue)
            for y in range(masked_hue.shape[0]):
                for x in range(masked_hue.shape[1]):
                    hue_value = masked_hue[y, x]
                    if hue_value is np.ma.masked:
                        scaled_hue[y, x] = np.ma.masked
                    elif hue_value >= lower_bound_1 and hue_value <= upper_bound_1:
                        scaled_hue[y, x] = hue_value - lower_bound_1  # Scale 280-360 to 0-80
                    elif hue_value <= upper_bound_2:
                        scaled_hue[y, x] = hue_value + (upper_bound_1 - lower_bound_1)  # Scale 0-90 to 80-170
                    else:
                        scaled_hue[y, x] = np.ma.masked
            
            mean_scaled_hue = scaled_hue.mean()

        # Append the calculated hue value for this tube
        hue_values.append(mean_scaled_hue)

    # Update the hue values using the moving average filter
    filtered_hues = update_hue_and_calculate_average(hue_values)

    # Apply the fake hue filter to clean the results
    #filtered_hues = filter_fake_hue(filtered_hues)

    # Draw rectangle and text on the image, and store the results
    for i, (tube, (x1, y1, x2, y2)) in enumerate(regions.items()):
        hue = filtered_hues[i]

        # Draw rectangle around the tube area
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Display the hue value or an empty space if None
        if hue is not None:
            cv2.putText(image, f"H: {hue:.0f}", (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            cv2.putText(image, "H: ", (x1 - 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Save the hue result for this tube
        results[tube] = {"hue": hue}

    return results


def capture_image_from_camera(output_path='captured_image.jpg'):
    try:
        # Construct the raspistill command
        command = [
            'raspistill',
            '-o', output_path,            
            '-w', '1280',
            '-h', '960',
            '-q', '80',
            '-t', '1000',
            '-hf','-vf',
            '-ss','25000',
            '-awb','auto',
            '-ISO','400',
            '-sa','0'
            #'-sh','70'
            #'-ifx','denoise' # 2 seconds delay before capture
        ]

        # Use subprocess.Popen for better control
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)  # Wait for up to 10 seconds

        if process.returncode != 0:
            # Print stderr for debugging purposes
            print(f"raspistill error output: {stderr.decode('utf-8')}")
            return None

        # Read the captured image using OpenCV
        image = cv2.imread(output_path)
        if image is None:
            raise ValueError("Failed to load the image")

        print(f"Image captured successfully and saved as {output_path}")
        return image
    except subprocess.TimeoutExpired:
        process.kill()  # Kill the process if timeout occurs
        print("Timeout: raspistill process killed.")
        return None
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None
# Thêm biến global để theo dõi số lần liên tiếp tất cả các tube T đều dương tính

positive_consecutive_count = 0  # Đếm số lần liên tiếp tất cả các tube đều dương tính


def program_1_at_t1(hue_i, hue_p, hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time, positive_consecutive_count, total_status, current_status
    table_data = []
    
    c1 = hue_i / hue_n if hue_i is not None else 0
    c2 = hue_p / hue_n if hue_p is not None else 0
    all_positive = True  # Biến cờ để kiểm tra tất cả tube T có dương tính hay không

    # Add Tube 1 (N) hue value
    table_data.append({
        "Tube": "Tube 1 (N)",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })
    # Adding Tube I and Tube P to the table data
    table_data.insert(1, {"Tube": "Tube 2 (I)", "Hue Value": hue_i, "C Value": c1, "Result": ""})
    table_data.insert(2, {"Tube": "Tube 3 (P)", "Hue Value": hue_p, "C Value": c2, "Result": ""})

    # Process the other test tubes
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None and hue_n != 0 else None

        # Apply new condition: hue > 110 and c_value >= COMPARATOR for positive result
        if hue_t > 110 and c_value is not None and c_value >= COMPARATOR:
            result = "Dương tính"
        # Apply new condition: hue < 110 and c_value < COMPARATOR for negative result
        elif hue_t < 110 and c_value is not None and c_value < COMPARATOR:
            result = "Âm tính"
            all_positive = False  # Nếu có tube âm tính, không được tính là tất cả đều dương tính
        else:
            result = ""
            all_positive = False  # Nếu không rõ kết quả thì cũng không tính là tất cả dương tính

        table_data.append({
            "Tube": f"Tube T{i+3}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    # Kiểm tra nếu tất cả các tube T đều dương tính
    if all_positive:
        positive_consecutive_count += 1
    else:
        positive_consecutive_count = 0  # Reset nếu có tube không dương tính

    # Nếu tất cả tube T dương tính 3 lần liên tiếp, dừng chương trình
    if positive_consecutive_count >= 3:
        program_trigger = False  # Dừng chương trình
        start_time = 0
        elapsed_time = 0
        positive_consecutive_count = 0  # Reset bộ đếm
        total_status = "Phản ứng đã dừng, tất cả tube T đều dương tính"
        current_status = "Phản ứng kết thúc"
        print("Dừng phản ứng: Tất cả tube T đều dương tính 3 lần liên tiếp")

    # Determine the total result based on C1 and C2
    if c1 is not None and c2 is not None:
        total_result = "Tiếp tục phản ứng" if c1 < COMPARATOR or c2 < COMPARATOR else "Thao tác tốt"
    else:
        total_result = "Dữ liệu không đầy đủ."
    
    return {
        "total_result": total_result,
        "table_data": table_data
    }

def program_2_at_t1(hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time, positive_consecutive_count, total_status, current_status
    table_data = []
    all_positive = True  # Biến cờ để kiểm tra tất cả tube T có dương tính hay không

    # Add Tube 1 (N) hue value
    table_data.append({
        "Tube": "Tube 1 (N)",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })

    # Process tubes T1 to T7
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None

        # Apply new condition: hue > 110 and c_value >= COMPARATOR for positive result
        if hue_t > 110 and c_value is not None and c_value >= COMPARATOR:
            result = "Dương tính"
        # Apply new condition: hue < 110 and c_value < COMPARATOR for negative result
        elif hue_t < 110 and c_value is not None and c_value < COMPARATOR:
            result = "Âm tính"
            all_positive = False  # Nếu có tube âm tính, không được tính là tất cả đều dương tính
        else:
            result = ""
            all_positive = False  # Nếu không rõ kết quả thì cũng không tính là tất cả dương tính

        table_data.append({
            "Tube": f"Tube T{i}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    # Kiểm tra nếu tất cả các tube T đều dương tính
    if all_positive:
        positive_consecutive_count += 1
    else:
        positive_consecutive_count = 0  # Reset nếu có tube không dương tính

    # Nếu tất cả tube T dương tính 3 lần liên tiếp, dừng chương trình
    if positive_consecutive_count >= 3:
        program_trigger = False  # Dừng chương trình
        start_time = 0
        elapsed_time = 0
        positive_consecutive_count = 0  # Reset bộ đếm
        total_status = "Phản ứng đã dừng, tất cả tube T đều dương tính"
        current_status = "Phản ứng kết thúc"
        print("Dừng phản ứng: Tất cả tube T đều dương tính 3 lần liên tiếp")

    return {
        "total_result": "Chương trình 2 kết thúc",
        "table_data": table_data
    }

def program_1_at_end(hue_i, hue_p, hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time
   
    c1 = hue_i / hue_n if hue_i is not None else 0
    c2 = hue_p / hue_n if hue_p is not None else 0
    c_values = {}
    table_data = []

    # Add Tube 1 (N) hue value
    table_data.append({
        "Tube": "Tube 1 (N)",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })
    if c1 >= COMPARATOR and hue_i > 110:
    # Adding Tube I and Tube P to the table data
        table_data.insert(1, {"Tube": "Tube 2 (I)", "Hue Value": hue_i, "C Value": c1, "Result": "Đạt"})
    else:
        table_data.insert(1, {"Tube": "Tube 2 (I)", "Hue Value": hue_i, "C Value": c1, "Result": "Không đạt"})
    if c2 >= COMPARATOR and hue_p > 110:
        table_data.insert(2, {"Tube": "Tube 3 (P)", "Hue Value": hue_p, "C Value": c2, "Result": "Đạt"})
    else:
        table_data.insert(2, {"Tube": "Tube 3 (P)", "Hue Value": hue_p, "C Value": c2, "Result": "Không đạt"})
    
    # Process the other test tubes
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None

        # Apply new condition: hue > 110 and c_value >= COMPARATOR for positive result
        if hue_t > 110 and c_value is not None and c_value >= COMPARATOR:
            result = "Dương tính"
        # Apply new condition: hue < 110 and c_value < COMPARATOR for negative result
        elif hue_t < 110 and c_value is not None and c_value < COMPARATOR:
            result = "Âm tính"
        else:
            result = ""

        table_data.append({
            "Tube": f"Tube T{i+3}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    # Determine the total result based on C1 and C2
    if c1 is not None and c2 is not None:
        total_result = "Thao tác không đạt. Kết thúc phản ứng" if c1 < COMPARATOR or c2 < COMPARATOR else "Thao tác tốt. Kết thúc phản ứng"
    else:
        total_result = "Dữ liệu không đầy đủ, tube 1 hoặc 2 không có. Kết thúc phản ứng"
    return {
        "total_result": total_result,
        "table_data": table_data
    }

def program_2_at_end(hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time

    table_data = []

    # Add Tube 1 (N) hue value
    table_data.append({
        "Tube": "Tube 1 (N)",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })

    # Process tubes T1 to T7
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None

        # Apply new condition: hue > 110 and c_value >= COMPARATOR for positive result
        if hue_t > 110 and c_value is not None and c_value >= COMPARATOR:
            result = "Dương tính"
        # Apply new condition: hue < 110 and c_value < COMPARATOR for negative result
        elif hue_t < 110 and c_value is not None and c_value < COMPARATOR:
            result = "Âm tính"
        else:
            result = ""

        table_data.append({
            "Tube": f"Tube T{i}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    return {
        "total_result": "Chương trình 2 kết thúc",
        "table_data": table_data
    }

def log_program_result_to_csv(program_result):
    global df_program

    for row in program_result['table_data']:
        new_row = {
            'Timestamp': datetime.datetime.now(),
            'Program': selected_program,
            'Tube': row['Tube'],
            'Hue Value':row['Hue Value'],
            'C Value': row['C Value'],
            'Result of Tube': row['Result']
        }

        # Convert new_row to a DataFrame
        new_df = pd.DataFrame([new_row])

        # Fill any NaN values with 0
        new_df = new_df.fillna(0)

        # Concatenate the new DataFrame with the main DataFrame
        df_program = pd.concat([df_program, new_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df_program.to_csv(program_csv_file, index=False)

def update_hue_n_and_check(hue_n):
    global last_five_hues_n
    
    # Add the new hue value to the list (hue_n could be None)
    last_five_hues_n.append(hue_n)
    
    # Keep only the last 5 hue values
    if len(last_five_hues_n) > 5:
        last_five_hues_n.pop(0)

    # Wait until at least 5 samples have been collected before making a conclusion
    if len(last_five_hues_n) < 5:
        return False  # Not enough data yet

    # Check if the last 5 hue values are either > 140 or None
    if all(hue is None or hue > 140 for hue in last_five_hues_n):
        return True  # Condition met: stop the program

    return False


def capture_and_save():
    global current_status, latest_image_path, capture_interval, program_trigger, program_result, elapsed_time, start_time, selected_process_time, selected_program, selected_t1
    start_time = 0
    capture_counter = 0
    capture_interval_seconds = 15  # Time in seconds between each capture
    t1_interval_counter = 0

    while not stop_event.is_set():
        pause_event.wait()
        
        try:
            if program_trigger is True:
                if capture_counter >= capture_interval_seconds:
                    capture_counter = 0  # Reset the counter after capturing

                    # Always capture and process the image, regardless of whether a program is triggered
                    image = capture_image_from_camera()
                    if image is None:
                        raise ValueError("Failed to capture image")
                    image = image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
                    
                    hue_value = detect_test_tube(image)
                    if hue_value is None:
                        raise ValueError("Failed to detect test tube hues")
                    
                    # Save the hue values and image
                    timestamp = datetime.datetime.now()
                    row_hue = [timestamp] + [hue_value[f'tube_{i}']["hue"] for i in range(1, 9)]
                    df_hue.loc[len(df_hue)] = row_hue
                    df_hue.to_csv(hue_csv_file, index=False)

                    latest_image_path = os.path.join(image_dir, f'test_tube_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg')
                    cv2.imwrite(latest_image_path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"Saved image: {latest_image_path}")

                    hue_n = hue_value['tube_1']['hue']  # Assuming tube_1 is Tube 1 (N)

                    if update_hue_n_and_check(hue_n):
                        table_data = []

                        # Add Tube 1 (N) hue value
                        table_data.append({
                            "Tube": "Tube 1 (N)",
                            "Hue Value": hue_n,
                            "C Value": "",
                            "Result": ""
                        })
                        program_result = {
                            'total_result': "Mẫu chứng âm không đạt. Dừng phản ứng",
                            'table_data': table_data
                        }
                        print(f"Program stopped: Hue N condition met after 5 cycles, Hue N: {hue_n}")
                        program_trigger = False
                        elapsed_time = 0
                        start_time = 0
                        continue  # Exit the loop

                    
                    if selected_program == 1:
                        program_return = program_1_at_t1(
                            hue_value['tube_2']['hue'],  # tube_I
                            hue_value['tube_3']['hue'],  # tube_P
                            hue_value['tube_1']['hue'],  # tube_N
                            [hue_value[f'tube_{i}']['hue'] for i in range(4, 9)]  # tubes T4 to T8
                        )
                        program_result = {
                            'total_result': program_return['total_result'],
                            'table_data': program_return['table_data']
                        }
                    elif selected_program == 2:
                        program_return = program_2_at_t1(
                            hue_value['tube_1']['hue'],  # tube_N
                            [hue_value[f'tube_{i}']['hue'] for i in range(2, 9)]  # tubes T1 to T7
                        )
                        program_result = {
                            'total_result': "Chương trình 2 đang chạy",
                            'table_data': program_return['table_data']
                        }
                
                else:
                    capture_counter += 1  # Increment the counter

                
                if start_time == 0:
                    start_time = time.time()  # Start counting from when the program is triggered
                elapsed_time = time.time() - start_time
                print(f'Thời gian chạy: {elapsed_time}')
                current_status = "Đang chạy"
                # Check if the process time is reached
                if elapsed_time >= selected_process_time:
                    # Handle end of process time event
                    if selected_program == 1:
                        program_return = program_1_at_end(
                            hue_value['tube_2']['hue'],  # tube_I
                            hue_value['tube_3']['hue'],  # tube_P
                            hue_value['tube_1']['hue'],  # tube_N
                            [hue_value[f'tube_{i}']['hue'] for i in range(4, 9)]  # tubes T4 to T8
                        )
                        program_result = {
                            'total_result': program_return['total_result'],
                            'table_data': program_return['table_data']
                        }
                    elif selected_program == 2:
                        program_return = program_2_at_end(
                            hue_value['tube_1']['hue'],  # tube_N
                            [hue_value[f'tube_{i}']['hue'] for i in range(2, 9)]  # tubes T1 to T7
                        )
                        program_result = {
                            'total_result': "Chương trình 2 đã kết thúc",
                            'table_data': program_return['table_data']
                        }

                    current_status = "Chương trình kết thúc"
                    log_program_result_to_csv(program_result)

                    # Disable the trigger after the process is complete
                    program_trigger = False
                    start_time = 0
                    elapsed_time = 0

                t1_interval_counter += 1  # Track time for T1 intervals
                # Log to CSV at T1 intervals
                if t1_interval_counter >= selected_t1:
                    log_program_result_to_csv(program_result)
                    t1_interval_counter = 0  # Reset T1 interval counter 
                
                
                sleep(capture_interval)
            
            else: #program trigger is false

                sleep(5)  # Increase sleep time if no program is running
                current_status = "Đang chờ"
                
            

        except Exception as e:
            # Handle any errors and prevent thread from stopping
            current_status = "Chương trình gặp lỗi, hãy thử lại"
            print(f"Error in capture thread: {e}")
            sleep(5)  # Retry after a delay




# Function to start capture thread
def start_capture_thread():
    global capture_thread, stop_event, df_hue, df_program

    stop_event.clear()

    # # Remove and recreate the hue CSV file if it exists
    if os.path.exists(hue_csv_file):
        os.remove(hue_csv_file)
    df_hue = pd.DataFrame(columns=hue_columns)
    df_hue.to_csv(hue_csv_file, index=False)

    # Remove and recreate the program results CSV file if it exists
    if os.path.exists(program_csv_file):
        os.remove(program_csv_file)
    df_program = pd.DataFrame(columns=program_columns)
    df_program.to_csv(program_csv_file, index=False)

    # # Ensure the image directory is clear
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Start the capture thread
    capture_thread = Thread(target=capture_and_save)
    capture_thread.daemon = True
    capture_thread.start()

def stop_capture_thread():
    stop_event.set()
    create_zip_backup()
    # Clear the dataframes and save to CSV
    df_hue = pd.DataFrame(columns=hue_columns)
    df_hue.to_csv(hue_csv_file, index=False)

    df_program = pd.DataFrame(columns=program_columns)
    df_program.to_csv(program_csv_file, index=False)

    # # Clear the image directory
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def plot_graph(columns=2):
    try:
        df = pd.read_csv(hue_csv_file)
    except pd.errors.EmptyDataError:
        return None  # Return None if the CSV file is empty

    if df.empty:
        return None  # No data available for plotting

    rows = int(np.ceil(8 / columns))  # Calculate the number of rows
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=[f'Tube {i}' for i in range(1, 9)])
    
    for i in range(1, 9):
        if f'Tube_{i}_Hue' in df.columns:
            row = (i - 1) // columns + 1
            col = (i - 1) % columns + 1
            fig.add_trace(
                go.Scatter(
                    x=df['Timestamp'], 
                    y=df[f'Tube_{i}_Hue'], 
                    mode='lines', 
                    name=f'Tube_{i}_Hue', 
                    line=dict(color='orange')
                ), 
                row=row, col=col
            )
    
    # Update layout with annotations adjusted to prevent overlap
    annotations = []
    for i, annotation in enumerate(fig['layout']['annotations']):
        annotations.append(
            dict(
                text=annotation['text'],
                x=annotation['x'] - (0.2 / columns),  # Adjust the x position based on columns
                y=annotation['y'],  # Adjust the y position to prevent overlap
                xref='paper',
                yref='paper',
                showarrow=False,
                align='left',  # Ensure text is aligned left
                xanchor='left'  # Anchor text to the left
            )
        )
    
    fig.update_layout(
        height=rows * 300,
        showlegend=False,
        annotations=annotations,
        margin=dict(l=5, r=5, t=50, b=10),  # Adjust margins for mobile view
        autosize=True,  # Let the plot automatically size itself
        plot_bgcolor='lightgrey',  # Set the plot background color to light grey
        paper_bgcolor='lightgrey',  # Set the paper (outside plot area) background color to grey
    )
    
    # Return data and layout separately
    return {"data": fig['data'], "layout": fig['layout']}

    
def handle_temperature(action, value=None):
    global serial_lock

    if action == 'get':
        # Use the lock for serial communication
        with serial_lock:
            ser.write(b'get_data\n')
            line = ser.readline().decode('utf-8').strip()

        if "Temperature" in line:
            # Extract data using string manipulation outside the lock
            try:
                data_list = line.split(':')  # Split on ':' delimiter
                temperature = float(data_list[1].split()[0])  # Extract temperature
                setpoint = float(data_list[2].split()[0])  # Extract setpoint
                output = float(data_list[3].split()[0])  # Extract output
                return temperature, setpoint, output
            except (IndexError, ValueError):
                # Return None for all values if there is an error in data extraction
                return None, None, None
        else:
            return None, None, None

    elif action == 'set' and value is not None:
        # Get the current temperature before locking the serial communication for 'set'
        temperature, _, _ = handle_temperature('get')

        if temperature is not None and temperature < 30:
            # Use the lock for the serial 'trigger' command
            with serial_lock:
                ser.write(b'trigger\n')
                response_trigger = ser.readline().decode('utf-8').strip()
                print(f"Trigger response: {response_trigger}")

        # Use the lock for setting the new setpoint
        with serial_lock:
            command = f'setpoint {value}\n'
            ser.write(command.encode())
            response_setpoint = ser.readline().decode('utf-8').strip()

        return response_setpoint

    else:
        return None, None, None

def create_zip_backup():
    # Get the current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define the zip file name with the timestamp
    zip_filename = f"backup_{timestamp}.zip"
    zip_filepath = os.path.join("/home/lamp/testtubenew/", zip_filename)  # Make sure to specify the correct backup directory

    # Create an in-memory zip file and add CSV and image files
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the hue CSV file
        zipf.write(hue_csv_file, arcname=f'test_tube_hue_values_{timestamp}.csv')
        
        # Add the program results CSV file
        zipf.write(program_csv_file, arcname=f'program_results_{timestamp}.csv')
        
        # Add all images from the image directory
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, image_dir))
    
    print(f"Data and images zipped into: {zip_filepath}")


app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/plot/<int:columns>')
def plot(columns):
    fig = plot_graph(columns)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify(graphJSON)

@app.route('/start')
def start():
    start_capture_thread()
    return jsonify({'status': 'started'})

@app.route('/pause')
def pause():
    pause_event.clear()
    return jsonify({'status': 'paused'})

@app.route('/resume')
def resume():
    pause_event.set()
    return jsonify({'status': 'resumed'})

@app.route('/reset')
def reset():
    global program_status,program_trigger, program_result, start_time, selected_process_time, selected_program, selected_t1, elapsed_time
    
    stop_capture_thread()
    # start_capture_thread()
    program_trigger = False
    selected_program = None
    program_status = None
    selected_t1 = None
    selected_process_time = None
    elapsed_time = 0
    start_time = 0
    program_result = {
                        'total_result': [],
                        'table_data': []}
    return jsonify({'status': 'reset'})

@app.route('/latest_image')
def latest_image():
    global latest_image_path
    print(f"Image path {latest_image_path}")
    if latest_image_path and os.path.exists(latest_image_path):
        return send_file(latest_image_path, mimetype='image/jpeg')
    else:
        return jsonify({'error': 'No image available'})

@app.route('/download_CSV')
def download_CSV():
    # Create an in-memory zip file
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add the hue values CSV to the zip file
        zipf.write(hue_csv_file, arcname='test_tube_hue_values.csv')
        # Add the program results CSV to the zip file
        zipf.write(program_csv_file, arcname='program_results.csv')
    
    # Move the pointer to the beginning of the in-memory zip file
    zip_buffer.seek(0)
    
    # Send the zip file to the user
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        attachment_filename='test_tube_data.zip'
    )

@app.route('/download')
def download():
    # Get today's date in the format YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, image_dir))
    zip_buffer.seek(0)
    # Name the zip file with today's date appended
    zip_filename = f"images_{today}.zip"
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, attachment_filename=zip_filename)

@app.route('/set_interval', methods=['POST'])
def set_interval():
    global capture_interval
    interval = request.form.get('interval', type=int)
    if interval and interval > 0:
        capture_interval = interval
        return jsonify({'status': 'interval set', 'interval': capture_interval})
    else:
        return jsonify({'error': 'Invalid interval'})

@app.route('/temperature')
def temperature():
  temperature, setpoint, output = handle_temperature('get')  # Unpack returned values
  if temperature is not None:  # Check if data is available
    return jsonify({
      'temperature': temperature,
      'setpoint': setpoint,
      'output': output
    })
  else:
    return jsonify({'message': 'No data available'})

@app.route('/set_temperature', methods=['POST'])
def set_temperature_route():
    value = request.form.get('value')
    if value:
        response = handle_temperature('set', value)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Invalid request'})

# Initialize the status
current_status = "Đang chờ"

@app.route('/status', methods=['POST'])
def set_status():
    global current_status, program_status
    data = request.get_json()

    if 'status' in data:
        current_status = data['status']

        # Include the current program status details if they exist
        if program_status:
            program_status['status'] = current_status
        else:
            program_status = {
                'status': current_status,
                'program': None,
                'check_time_t1': None,
                'process_time': None,
                'temperature': None
            }

        return jsonify({'status': 'success'}), 200
    return jsonify({'status': 'error'}), 400


@app.route('/status', methods=['GET'])
def get_status():
    global current_status, program_trigger

    if program_trigger:
        return jsonify({
            'status': current_status,
            'program': selected_program,
            'check_time_t1': selected_t1,
            'process_time': selected_process_time,
            'temperature': selected_temperature
        })
    else:
        return jsonify({
            'status': "Đang chờ",
            'program': None,
            'check_time_t1': None,
            'process_time': None,
            'temperature': None
        })

@app.route('/run_program', methods=['POST'])
def run_program():
    global program_trigger,selected_program, selected_temperature, selected_t1, selected_process_time, program_status

    try:
        data = request.get_json()
        selected_program = int(data['program'])
        selected_temperature = float(data['temperature'])
        selected_t1 = int(data['checkTime'])
        selected_process_time = int(data['processTime'])
        program_trigger = True  # Enable the trigger for the capture_and_save function
        if selected_temperature < 0 or selected_process_time <= 0 or selected_t1 <= 0:
            raise ValueError("Invalid input values")

        # Do not trigger the program yet, just set parameters
        handle_temperature('set', selected_temperature)

        # Update program status but do not start the program yet
        program_status = {
            'status': 'Program setup complete',
            'program': selected_program,
            'check_time_t1': selected_t1,
            'process_time': selected_process_time,
            'temperature': selected_temperature
        }

        # Add more program names for better flexibility
        program_names = {
            1: "Làm quen quy trình",
            2: "Chạy mẫu thử"
        }

        program_result = {
            'status': 'success',
            'program_name': program_names.get(selected_program, "Unknown Program"),
            'message': f'Program {selected_program} parameters have been set'
        }

        return jsonify(program_result)

    except (ValueError, KeyError) as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }), 500

@app.route('/get_program_result', methods=['GET'])
def get_program_result():
    global program_result

    if program_result:
        return jsonify(program_result)
    else:
        return jsonify({'result': 'No result available yet'})

@app.route('/elapsed_time', methods=['GET'])
def get_elapsed_time():
    global elapsed_time
    return jsonify({'elapsed_time': elapsed_time})

@app.route('/program_trigger', methods=['GET'])
def get_program_trigger():
    global program_trigger
    return jsonify({'program_trigger': program_trigger})

@app.route('/fetch_all_data', methods=['GET'])
def fetch_all_data():
    global selected_process_time
    # Fetch temperature data
    temperature, setpoint, output = handle_temperature('get')
    
    # Fetch the program result
    program_result = get_program_result()
    
    # Fetch elapsed time
    elapsed_time = get_elapsed_time()

    # Fetch program_trigger
    program_trigger = get_program_trigger()

    # Fetch the latest image URL (you might need to adjust this depending on how the image is served)
    image_url = '/latest_image?' + str(time.time())  # Add timestamp to prevent caching

    # Fetch plot data
    columns = 2  # Or however you decide to configure the number of columns
    fig = plot_graph(columns)
    
    # Serialize plot data using Plotly's JSON encoder
    plot_data = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    
    # Prepare the response
    response_data = {
        'temperature': {
            'temperature': temperature,
            'setpoint': setpoint,
            'output': output
        },
        'program_result': program_result.json,
        'elapsed_time': elapsed_time.json['elapsed_time'],
        'program_trigger': program_trigger.json['program_trigger'],
        'process_time': selected_process_time,
        'image_url': image_url,
        'plot_data': plot_data
    }

    return jsonify(response_data)

@app.route('/setup_wifi', methods=['POST'])
def setup_wifi():
    try:
        data = request.get_json()
        ssid_new = data['ssid']
        password_new = data['password']

        # Kiểm tra và khởi động NetworkManager nếu cần
        nm_status = subprocess.run(['systemctl', 'is-active', 'NetworkManager'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if nm_status.stdout.decode().strip() != 'active':
            print("NetworkManager không chạy. Đang khởi động NetworkManager...")
            subprocess.run(['sudo', 'systemctl', 'start', 'NetworkManager'], check=True)
            subprocess.run(['sudo', 'systemctl', 'enable', 'NetworkManager'], check=True)
            time.sleep(10)  # Đợi một chút để NetworkManager khởi động

        # Kiểm tra Wi-Fi hiện tại
        current_wifi = subprocess.run(['nmcli', '-t', '-f', 'active,ssid', 'dev', 'wifi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_wifi_list = current_wifi.stdout.decode().strip().split('\n')
        current_ssid = None
        for wifi in current_wifi_list:
            if wifi.startswith('yes:'):
                current_ssid = wifi.split(':')[1]
                break

        if current_ssid == ssid_new:
            print(f"Bạn đã kết nối với Wi-Fi {ssid_new} rồi.")
            return jsonify({'status': 'error', 'message': f'Bạn đã kết nối với Wi-Fi {ssid_new} rồi.'})

        # Kết nối tới mạng Wi-Fi mới
        print(f"Đang kết nối tới Wi-Fi mới: SSID={ssid_new}, PASSWORD={password_new}")
        result = subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'connect', ssid_new, 'password', password_new], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"nmcli stdout: {result.stdout.decode().strip()}")
        print(f"nmcli stderr: {result.stderr.decode().strip()}")
        if result.returncode == 0:
            print("Kết nối thành công tới Wi-Fi mới")
            return jsonify({'status': 'success', 'message': 'Kết nối thành công tới Wi-Fi mới', 'connected_ssid': ssid_new})
        else:
            print(f"Kết nối tới Wi-Fi mới thất bại: {result.stderr.decode().strip()}")
            # Attempt to reconnect to the previous Wi-Fi network
            if current_ssid:
                print(f"Đang kết nối lại tới Wi-Fi cũ: SSID={current_ssid}")
                reconnect_result = subprocess.run(['sudo', 'nmcli', 'connection', 'up', current_ssid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if reconnect_result.returncode == 0:
                    print(f"Kết nối lại thành công tới Wi-Fi cũ: {current_ssid}")
                    return jsonify({'status': 'error', 'message': f'Kết nối tới Wi-Fi mới thất bại, đã kết nối lại tới Wi-Fi cũ: {current_ssid}', 'error': result.stderr.decode().strip()})
                else:
                    print(f"Kết nối lại tới Wi-Fi cũ thất bại: {reconnect_result.stderr.decode().strip()}")
                    return jsonify({'status': 'error', 'message': f'Kết nối tới Wi-Fi mới thất bại và không thể kết nối lại tới Wi-Fi cũ: {current_ssid}', 'error': result.stderr.decode().strip()})
            else:
                return jsonify({'status': 'error', 'message': 'Kết nối tới Wi-Fi mới thất bại và không có Wi-Fi cũ để kết nối lại', 'error': result.stderr.decode().strip()})

    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/scan_wifi', methods=['GET'])
def scan_wifi():
    try:
        # Run the rescan command
        subprocess.run(['sudo', 'nmcli', 'dev', 'wifi', 'rescan'], check=True)

        # Get the list of available Wi-Fi networks
        result = subprocess.run(['sudo', 'nmcli', '-t', '-f', 'SSID,SECURITY', 'dev', 'wifi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()

        if result.returncode != 0:
            return jsonify({'status': 'error', 'message': 'Quét mạng Wi-Fi thất bại', 'error': result.stderr.decode('utf-8').strip()}), 500

        # Get the currently connected Wi-Fi network
        current_wifi_result = subprocess.run(['sudo', 'nmcli', '-t', '-f', 'active,ssid', 'dev', 'wifi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        current_wifi_output = current_wifi_result.stdout.decode('utf-8').strip()
        current_ssid = None
        for line in current_wifi_output.split('\n'):
            if line.startswith('yes:'):
                current_ssid = line.split(':')[1]
                break

        # Parse the output into a list of dictionaries
        wifi_list = []
        for line in output.split('\n'):
            ssid, security = line.split(':')
            wifi_list.append({'ssid': ssid, 'security': security})

        # Reorder the list to place the currently connected Wi-Fi at index 0
        if current_ssid:
            wifi_list = sorted(wifi_list, key=lambda x: x['ssid'] != current_ssid)

        return jsonify({'status': 'success', 'wifi_list': wifi_list})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
if __name__ == '__main__':
    handle_temperature('set', 25)
    
    app.run(host='0.0.0.0', port=80)