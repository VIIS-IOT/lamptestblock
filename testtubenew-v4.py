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

COMPARATOR = 1.4
ORANGE_OFFSET = 20

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


# Function to convert hue to pH
def hue_to_ph(hue):
    if 45 <= hue < 75:
        return 6.0 + (hue - 45) * (0.6 / 30)  # Interpolating within the yellow range
    elif 30 <= hue < 45:
        return 6.7 + (hue - 30) * (0.4 / 15)  # Interpolating within the orange range
    elif (0 <= hue < 30) or (330 <= hue < 360):
        if hue < 30:
            return 7.2 + (hue - 0) * (0.4 / 30)  # Interpolating within the red range (low end)
        else:
            return 7.2 + (hue - 330) * (0.4 / 30)  # Interpolating within the red range (high end)
    elif 300 <= hue < 330:
        return 7.7 + (hue - 300) * (0.3 / 30)  # Interpolating within the pink/magenta range
    else:
        return None  # If the hue doesn't fall within any expected range, return None

# Initialize the Kalman filter for 8 test tubes
kf = [KalmanFilter(initial_state_mean=0, n_dim_obs=1, transition_matrices=1, observation_matrices=1, initial_state_covariance=1,transition_covariance=1e-3,observation_covariance=1e-1) for _ in range(8)]
state_means = [np.array([0]) for _ in range(8)]
state_covariances = [np.array([[1]]) for _ in range(8)]

def detect_test_tube(image):
    results = {}
    global state_means, state_covariances
    valid_pixel_threshold = 20  # Minimum number of valid pixels required
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hue_channel = hsv_image[:, :, 0]  # Ensure hue value is not divided by 2
    for i, (tube, (x1, y1, x2, y2)) in enumerate(regions.items()):
        sub_image = image[y1:y2, x1:x2]   
        hsv_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]  # Ensure hue value is not divided by 2            

        # Define the bounds
        lower_bound_1 = 280  # Directly use degrees, not scaled down
        upper_bound_1 = 360
        lower_bound_2 = 0
        upper_bound_2 = 90

        # Convert hue_channel to the correct scale (0-360)
        hue_channel = hue_channel.astype(float) * 2

        # Create mask to include only hues within the red-yellow range (280°-360° and 0°-90°)
        mask_hue = ((hue_channel >= lower_bound_1) | (hue_channel <= upper_bound_2)).astype(np.uint8)

        # Mask the hue values outside the desired range
        masked_hue = np.ma.masked_array(hue_channel, mask_hue == 0)
        # Filter out noise: Only proceed if there are enough valid pixels
        valid_pixel_count = np.ma.count(masked_hue)
        if valid_pixel_count < valid_pixel_threshold:
            mean_scaled_hue = None
        else:
        #     # Debugging: Print out the masked hue values
        #     # print(f"Masked hue values for {tube}: {masked_hue}")

        #     # Scale and convert hue values
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

            # Debugging: Print out the scaled hue values
            #print(f"Masked hue values for {tube}: {masked_hue}")
            #print(f"Scaled hue values for {tube}: {scaled_hue}")

            # Calculate mean of scaled hue values
            mean_scaled_hue = scaled_hue.mean()
            if mean_scaled_hue < 110 and mean_scaled_hue > 95:
                mean_scaled_hue = mean_scaled_hue - 20

            # mean_scaled_hue = masked_hue.mean()
            print(f"Mean hue values for {tube}: {mean_scaled_hue}")
        
        

        ph = hue_to_ph(mean_scaled_hue) if mean_scaled_hue is not None else None

        # Draw rectangle and text on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if mean_scaled_hue is not None:
            cv2.putText(image, f"H: {mean_scaled_hue:.0f}", (x1-5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            
        else:
            cv2.putText(image, "H: ", (x1-5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            

        results[tube] = {"hue": mean_scaled_hue, "ph": ph}

    return results

def capture_image_from_camera(output_path='captured_image.jpg'):
    try:
        # Construct the raspistill command
        command = [
            'raspistill',
            '-o', output_path,            
            '-w', '1280',
            '-h', '960',
            '-q', '100',
            '-t', '1000',
            '-hf','-vf',
            '-ss','14000',
            '-awb','auto'   # 2 seconds delay before capture
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

def program_1_at_t1(hue_i, hue_p, hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time
    # hue_i = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_p = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_n = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_t_list = [random.uniform(50, 150) for _ in range(5)]  # Simulate hue values for tubes T1 to T7
    if hue_n is None or hue_n == 0:
        return {
            "total_result": "Không có mẫu chứng âm. Kết thúc phản ứng.",
            "table_data": []
        }
     # Check if tube N hue is less than 95
    if hue_n > 100:
        
        table_data.append({
            "Tube": "Tube N",
            "Hue Value": hue_n,
            "C Value": "",
            "Result": ""
        })
        return {
            "total_result": "Mẫu chứng âm không đạt. Kết thúc phản ứng",
            "table_data": table_data
        }

    c1 = hue_i / hue_n if hue_i is not None else None
    c2 = hue_p / hue_n if hue_p is not None else None
    c_values = {}
    table_data = []

    # Add Tube N hue value
    table_data.append({
        "Tube": "Tube N",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })
    # Adding Tube I and Tube P to the table data
    table_data.insert(1, {"Tube": "Tube I", "Hue Value": hue_i, "C Value": c1, "Result": ""})
    table_data.insert(2, {"Tube": "Tube P", "Hue Value": hue_p, "C Value": c2, "Result": ""})


    # Process the other test tubes
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None
        c_values[f'C3{i}'] = c_value
        result = "Dương tính" if c_value is not None and c_value > COMPARATOR else "Âm tính"
        table_data.append({
            "Tube": f"Tube T{i+3}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    
    # Determine the total result based on C1 and C2
    total_result = "Tiếp tục phản ứng" if c1 < COMPARATOR or c2 < COMPARATOR else "Thao tác tốt"

    return {
        "total_result": total_result,
        "table_data": table_data
    }

def program_1_at_end(hue_i, hue_p, hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time
    # hue_i = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_p = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_n = random.uniform(50, 100)  # Simulate a hue value for tube N
    # hue_t_list = [random.uniform(50, 150) for _ in range(5)]  # Simulate hue values for tubes T1 to T7
    if hue_n is None or hue_n == 0:
        return {
            "total_result": "Không có mẫu chứng âm. Kết thúc phản ứng",
            "table_data": []
        }
     # Check if tube N hue is less than 95
    if hue_n > 100:
        
        table_data.append({
            "Tube": "Tube N",
            "Hue Value": hue_n,
            "C Value": "",
            "Result": ""
        })
        return {
            "total_result": "Mẫu chứng âm không đạt. Kết thúc phản ứng",
            "table_data": table_data
        }

    c1 = hue_i / hue_n if hue_i is not None else None
    c2 = hue_p / hue_n if hue_p is not None else None
    c_values = {}
    table_data = []

    # Add Tube N hue value
    table_data.append({
        "Tube": "Tube N",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })
    # Adding Tube I and Tube P to the table data
    table_data.insert(1, {"Tube": "Tube I", "Hue Value": hue_i, "C Value": c1, "Result": ""})
    table_data.insert(2, {"Tube": "Tube P", "Hue Value": hue_p, "C Value": c2, "Result": ""})

    # Process the other test tubes
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None
        c_values[f'C3{i}'] = c_value
        result = "Dương tính" if c_value is not None and c_value > COMPARATOR else "Âm tính"
        table_data.append({
            "Tube": f"Tube T{i+3}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })    

    # Determine the total result based on C1 and C2
    total_result = "Thao tác không đạt" if c1 < COMPARATOR or c2 < COMPARATOR else "Thao tác tốt"
    
    return {
        "total_result": total_result,
        "table_data": table_data
    }
   
def program_2_at_t1(hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time
    if hue_n is None or hue_n == 0:
        
        return {
            "total_result": "Không có mẫu chứng âm. Kết thúc phản ứng",
            "table_data": []
        }
     # Check if tube N hue is less than 95
    if hue_n > 100:
        
        table_data.append({
            "Tube": "Tube N",
            "Hue Value": hue_n,
            "C Value": "",
            "Result": ""
        })
        return {
            "total_result": "Mẫu chứng âm không đạt. Kết thúc phản ứng",
            "table_data": table_data
        }

    table_data = []

    # Add Tube N hue value
    table_data.append({
        "Tube": "Tube N",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })

    # Process tubes T1 to T7
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None
        result = "Dương tính" if c_value is not None and c_value > COMPARATOR else "Âm tính"
        table_data.append({
            "Tube": f"Tube T{i}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    # Determine the total result based on the values in the tubes
    # stop_flag = all(c_value <= COMPARATOR for c_value in [hue_t / hue_n for hue_t in hue_t_list if hue_n is not None])
    total_result =  "Tiếp tục phản ứng"

    return {
        "total_result": total_result,
        "table_data": table_data
    }


def program_2_at_end(hue_n, hue_t_list):
    global program_trigger, start_time, elapsed_time
    if hue_n is None or hue_n == 0:
        return {
            "total_result": "Invalid input: hue_n is None or zero.",
            "table_data": []
        }
    # Check if tube N hue is less than 95
    if hue_n > 100:
        
        table_data.append({
            "Tube": "Tube N",
            "Hue Value": hue_n,
            "C Value": "",
            "Result": ""
        })
        return {
            "total_result": "Mẫu chứng âm không đạt. Kết thúc phản ứng",
            "table_data": table_data
        }

    table_data = []

    # Add Tube N hue value
    table_data.append({
        "Tube": "Tube N",
        "Hue Value": hue_n,
        "C Value": "",
        "Result": ""
    })

    # Process tubes T1 to T7
    for i, hue_t in enumerate(hue_t_list, start=1):
        if hue_t is None:  # Skip this tube if hue_t is None
            continue
        c_value = hue_t / hue_n if hue_t is not None else None
        result = "Dương tính" if c_value is not None and c_value > COMPARATOR else "Âm tính"
        table_data.append({
            "Tube": f"Tube T{i}",
            "Hue Value": hue_t,
            "C Value": c_value,
            "Result": result
        })

    # Determine the total result based on tube values
    stop_flag = all(c_value <= COMPARATOR for c_value in [hue_t / hue_n for hue_t in hue_t_list if hue_n is not None])
    total_result = "Thao tác tốt" if stop_flag else "Tiếp tục phản ứng"

    return {
        "total_result": total_result,
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

        # Convert new_row to a DataFrame and append it to the main df_program
        new_df = pd.DataFrame([new_row])
        df_program = pd.concat([df_program, new_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV
    df_program.to_csv(program_csv_file, index=False)

def capture_and_save():
    global latest_image_path, capture_interval, program_trigger, program_result,elapsed_time, start_time, selected_process_time, selected_program, selected_t1
    start_time = 0
    capture_counter = 0
    capture_interval_seconds = 15  # Time in seconds between each capture
    t1_interval_counter = 0

    while not stop_event.is_set():
        pause_event.wait()
        
        if capture_counter >= capture_interval_seconds:
            capture_counter = 0  # Reset the counter after capturing            

            # Always capture and process the image, regardless of whether a program is triggered
            image = capture_image_from_camera()
            image = image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]
            if image is None:
                continue
            
            hue_value = detect_test_tube(image)
           
            
            # Save the hue values and image
            timestamp = datetime.datetime.now()
            row_hue = [timestamp] + [hue_value[f'tube_{i}']["hue"] for i in range(1, 9)]
            df_hue.loc[len(df_hue)] = row_hue
            df_hue.to_csv(hue_csv_file, index=False)

            latest_image_path = os.path.join(image_dir, f'test_tube_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg')
            cv2.imwrite(latest_image_path, image)
            print(f"Saved image: {latest_image_path}")      
            
           
            hue_n = hue_value['tube_1']['hue']  # Assuming tube_1 is Tube N

            # Check if tube N hue > 100, stop the program and show the result
            if hue_n is not None and hue_n > 100:
                table_data = []

                # Add Tube N hue value
                table_data.append({
                    "Tube": "Tube N",
                    "Hue Value": hue_n,
                    "C Value": "",
                    "Result": ""
                })
                program_result = {
                    'total_result': "Mẫu chứng âm không đạt vì HUE tube N > 100. Dừng phản ứng",
                    'table_data': table_data
                }
                print(f"Program stopped: Hue N > 100, Hue N: {hue_n}")
                # Stop the program
                program_trigger = False
                elapsed_time = 0
                start_time = 0
                continue  # Exit the function, which effectively stops the loop

            if program_trigger:
                
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
        if program_trigger:
            if start_time == 0:
                start_time = time.time()  # Start counting from when the program is triggered
            elapsed_time = time.time() - start_time
    
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
                # Log the final results to the CSV at the end of process time
                log_program_result_to_csv(program_result)

                # Send the final result to the web view
                print(f"Final result: {program_result}")
                
                # Disable the trigger after the process is complete
                program_trigger = False
                start_time = 0
                elapsed_time = 0

            t1_interval_counter += 1  # Track time for T1 intervals
            # Log to CSV at T1 intervals
            if t1_interval_counter >= selected_t1:
                log_program_result_to_csv(program_result)
                t1_interval_counter = 0  # Reset T1 interval counter 
        if not program_trigger:
            sleep(5)  # Increase sleep time if no program is running
        else:
            sleep(capture_interval)

# Function to start capture thread
def start_capture_thread():
    global capture_thread, stop_event, df_hue, df_program

    stop_event.clear()

    handle_temperature('set', 25)

    # Remove and recreate the hue CSV file if it exists
    if os.path.exists(hue_csv_file):
        os.remove(hue_csv_file)
    df_hue = pd.DataFrame(columns=hue_columns)
    df_hue.to_csv(hue_csv_file, index=False)

    # Remove and recreate the program results CSV file if it exists
    if os.path.exists(program_csv_file):
        os.remove(program_csv_file)
    df_program = pd.DataFrame(columns=program_columns)
    df_program.to_csv(program_csv_file, index=False)

    # Ensure the image directory is clear
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

    # Clear the dataframes and save to CSV
    df_hue = pd.DataFrame(columns=hue_columns)
    df_hue.to_csv(hue_csv_file, index=False)

    df_program = pd.DataFrame(columns=program_columns)
    df_program.to_csv(program_csv_file, index=False)

    # Clear the image directory
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
    if action == 'get':
        ser.write(b'get_data\n')
        line = ser.readline().decode('utf-8').strip()
        if "Temperature" in line:
            # Extract data using string manipulation
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
        # Get the current temperature
        temperature, _, _ = handle_temperature('get')
        
        if temperature is not None and temperature < 30:
            # If the temperature is below 30 degrees, send the 'trigger' command
            ser.write(b'trigger\n')
            response_trigger = ser.readline().decode('utf-8').strip()
            print(f"Trigger response: {response_trigger}")

        # Set the new setpoint
        command = f'setpoint {value}\n'
        ser.write(command.encode())
        response_setpoint = ser.readline().decode('utf-8').strip()
        return response_setpoint
    else:
        return None, None, None


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
    global program_trigger, program_result, start_time, selected_process_time, selected_program, selected_t1
    
    stop_capture_thread()
    # start_capture_thread()
    program_trigger = False
    selected_program = None
    
    selected_t1 = None
    selected_process_time = None
    
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
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(image_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, image_dir))
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, attachment_filename='images.zip')

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
current_status = "Đang chờ lệnh"

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
    global current_status, program_status

    if program_status:
        return jsonify({
            'status': current_status,
            'program': program_status['program'],
            'check_time_t1': program_status['check_time_t1'],
            'process_time': program_status['process_time'],
            'temperature': program_status['temperature']
        })
    else:
        return jsonify({
            'status': current_status,
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

@app.route('/fetch_all_data', methods=['GET'])
def fetch_all_data():
    # Fetch temperature data
    temperature, setpoint, output = handle_temperature('get')
    
    # Fetch the program result
    program_result = get_program_result()
    
    # Fetch elapsed time
    elapsed_time = get_elapsed_time()

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
        'image_url': image_url,
        'plot_data': plot_data
    }

    return jsonify(response_data)


if __name__ == '__main__':
    handle_temperature('set', 25)
    
    app.run(host='0.0.0.0', port=5000)
