import cv2
import os
import numpy as np
import subprocess

# Define the regions for the test tubes
regions = {
    "tube_1": (24, 70, 54, 100),
    "tube_2": (84, 70, 114, 100),
    "tube_3": (148, 70, 178, 100),
    "tube_4": (211, 70, 241, 100),
    "tube_5": (278, 70, 308, 100),
    "tube_6": (345, 70, 375, 100),
    "tube_7": (410, 70, 440, 100),
    "tube_8": (476, 70, 506, 105)
}

# Define Kalman filters for each tube (same as your setup)
from pykalman import KalmanFilter
kf = [KalmanFilter(initial_state_mean=0, n_dim_obs=1, 
                   transition_matrices=1, observation_matrices=1, 
                   initial_state_covariance=1, 
                   transition_covariance=1e-2,  
                   observation_covariance=5e-2)  
      for _ in range(8)]

state_means = [np.array([0]) for _ in range(8)]
state_covariances = [np.array([[1]]) for _ in range(8)]

# Function to capture an image using raspistill
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
            '-hf', '-vf',
            '-ss', '18000',
            '-awb', 'auto',
            '-ISO', '400',
            '-sa', '-20',
            '-sh', '50'
        ]

        # Use subprocess.Popen for better control
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=10)

        if process.returncode != 0:
            print(f"raspistill error output: {stderr.decode('utf-8')}")
            return None

        # Read the captured image using OpenCV
        image = cv2.imread(output_path)
        if image is None:
            raise ValueError("Failed to load the image")

        print(f"Image captured successfully and saved as {output_path}")
        return image
    except subprocess.TimeoutExpired:
        process.kill()
        print("Timeout: raspistill process killed.")
        return None
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

# Function to detect test tubes and save visualized sub-images
def detect_test_tube(image, output_dir="tube_images"):
    results = {}
    global state_means, state_covariances

    BRIGHTNESS_THRESHOLD = 240  # Define brightness threshold for filtering noise

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (tube, (x1, y1, x2, y2)) in enumerate(regions.items()):
        sub_image = image[y1:y2, x1:x2]
        blur_sub_image = cv2.GaussianBlur(sub_image, (3, 3), 0)
        hsv_image = cv2.cvtColor(blur_sub_image, cv2.COLOR_BGR2HSV)
        hue_channel = hsv_image[:, :, 0]
        value_channel = hsv_image[:, :, 2]

        # Convert the hue and value channels to grayscale for visualization
        hue_before_mask = cv2.normalize(hue_channel, None, 0, 255, cv2.NORM_MINMAX)
        value_gray = cv2.normalize(value_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Save the V (value) channel grayscale image
        value_path = os.path.join(output_dir, f'{tube}_value_channel.jpg')
        cv2.imwrite(value_path, value_gray)
        print(f"Saved value channel image for {tube} at {value_path}")

        # Masking the V channel (to filter out bright areas)
        mask_bright_pixels = value_channel < BRIGHTNESS_THRESHOLD
        hue_channel_filtered = np.ma.masked_array(hue_channel, mask=~mask_bright_pixels)

        # Convert hue_channel to the correct scale (0-360 degrees)
        hue_channel_filtered = hue_channel_filtered.astype(float) * 2

        # Save the hue before masking (for debugging purposes)
        hue_before_mask_path = os.path.join(output_dir, f'{tube}_hue_before_mask.jpg')
        cv2.imwrite(hue_before_mask_path, hue_before_mask)
        print(f"Saved hue before mask image for {tube} at {hue_before_mask_path}")

        # Filter hues in red-yellow range
        lower_bound_1 = 280
        upper_bound_1 = 360
        lower_bound_2 = 0
        upper_bound_2 = 90

        # Create a mask for the hue range
        mask_hue = ((hue_channel_filtered >= lower_bound_1) | (hue_channel_filtered <= upper_bound_2)).astype(np.uint8)

        # Apply morphological opening to remove small isolated noise
        kernel = np.ones((5, 5), np.uint8)  # Define a 3x3 kernel for morphological operations
        mask_hue = cv2.morphologyEx(mask_hue, cv2.MORPH_OPEN, kernel)

        # Apply the brightness mask and the filtered hue mask
        final_mask = np.logical_and(mask_hue, mask_bright_pixels)
        masked_hue = np.ma.masked_array(hue_channel_filtered, mask=~final_mask)

        valid_pixel_count = np.ma.count(masked_hue)
        if valid_pixel_count < 70:  # Minimum pixel threshold
            mean_scaled_hue = None
        else:
            # Scale hue values for the specified ranges
            scaled_hue = np.zeros_like(masked_hue)
            for y in range(masked_hue.shape[0]):
                for x in range(masked_hue.shape[1]):
                    hue_value = masked_hue[y, x]
                    if hue_value >= lower_bound_1 and hue_value <= upper_bound_1:
                        scaled_hue[y, x] = hue_value - lower_bound_1
                    elif hue_value <= upper_bound_2:
                        scaled_hue[y, x] = hue_value + (upper_bound_1 - lower_bound_1)
                    else:
                        scaled_hue[y, x] = np.ma.masked

            mean_scaled_hue = scaled_hue.mean()

            # Apply Kalman filter
            if mean_scaled_hue is not None:
                state_means[i], state_covariances[i] = kf[i].filter_update(
                    state_means[i], state_covariances[i], mean_scaled_hue)
                hue = state_means[i][0]
            else:
                hue = None

            # Convert the scaled hue to grayscale and save
            hue_after_mask = cv2.normalize(masked_hue.filled(0), None, 0, 255, cv2.NORM_MINMAX)
            hue_after_mask_path = os.path.join(output_dir, f'{tube}_hue_after_mask_scale.jpg')
            cv2.imwrite(hue_after_mask_path, hue_after_mask)
            print(f"Saved hue after mask and scaling image for {tube} at {hue_after_mask_path}")

        # Store the result
        results[tube] = {"hue": mean_scaled_hue}

        # Display the tube regions and hue on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if mean_scaled_hue is not None:
            cv2.putText(image, f"H: {mean_scaled_hue:.0f}", (x1-5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        else:
            cv2.putText(image, "H: ", (x1-5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    return results


# Main script to capture image and run detect_test_tube
if __name__ == '__main__':
    output_image_path = './test_tube_image.jpg'
    output_dir = './tube_images'

    # Capture the image
    image = capture_image_from_camera(output_path=output_image_path)
    if image is not None:
        # Crop the image if necessary (you can adjust these coordinates)
        CROP_Y1, CROP_Y2, CROP_X1, CROP_X2 = 308, 418, 337, 887
        image = image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

        # Run the test tube detection algorithm
        results = detect_test_tube(image, output_dir=output_dir)

        # Save the image with annotations
        cv2.imwrite(output_image_path, image)
        print(f"Results: {results}")
        print(f"Annotated image saved at {output_image_path}")
    else:
        print("Failed to capture image.")
