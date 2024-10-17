import cv2
import os
import numpy as np
import subprocess

# Define the regions for the test tubes

#Regions for each test tube in the image
regions = {
    "tube_1": (25, 55, 50, 72),
    "tube_2": (87, 55, 112, 72),
    "tube_3": (148, 52, 173, 70),
    "tube_4": (217, 48, 242, 70),
    "tube_5": (282, 47, 311, 66),
    "tube_6": (350, 47, 375, 66),
    "tube_7": (413, 55, 441, 72),
    "tube_8": (480, 55, 505, 72)
}
CROP_Y1 = 370
CROP_Y2 = 470
CROP_X1 = 290
CROP_X2 = 840
VALID_PIXEL_THRESHOLD = 20
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
            '-q', '100',
            '-t', '1000',
            '-hf','-vf',
            
            '-ss', '10000',
            '-awb', 'auto',
            '-ISO', '400',
            # '-sa', '0',
            #'-co','-10'
            # '-sh','40'
            #'-br','55'
            #'-ifx','denoise' # 2 seconds delay before capture
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
        
        # Count total pixels in the sub-image
        total_pixels = hue_channel.size

        # Convert the hue and value channels to grayscale for visualization
        value_gray = cv2.normalize(value_channel, None, 0, 255, cv2.NORM_MINMAX)

        # Save the V (value) channel grayscale image
        value_path = os.path.join(output_dir, f'{tube}_value_channel.jpg')
        cv2.imwrite(value_path, value_gray)

        # Masking the V channel (to filter out bright areas)
        mask_bright_pixels = value_channel < BRIGHTNESS_THRESHOLD
        hue_channel_filtered = np.ma.masked_array(hue_channel, mask=~mask_bright_pixels)

        # Convert hue_channel to the correct scale (0-360 degrees)
        hue_channel_filtered = hue_channel_filtered.astype(float) * 2

        # Filter hues in red-yellow range
        lower_bound_1 = 270
        upper_bound_1 = 360
        lower_bound_2 = 0
        upper_bound_2 = 90

        # Create a mask for the hue range
        mask_hue = ((hue_channel_filtered >= lower_bound_1) | (hue_channel_filtered <= upper_bound_2)).astype(np.uint8)

        # Apply morphological opening to remove small isolated noise
        kernel = np.ones((5, 5), np.uint8)  
        mask_hue = cv2.morphologyEx(mask_hue, cv2.MORPH_OPEN, kernel)
        mask_hue = cv2.morphologyEx(mask_hue, cv2.MORPH_CLOSE, kernel)
        # Apply the brightness mask and the filtered hue mask
        final_mask = np.logical_and(mask_hue, mask_bright_pixels)
        masked_hue = np.ma.masked_array(hue_channel_filtered, mask=~final_mask)

        # Count valid pixels in the hue range
        valid_pixel_count = np.ma.count(masked_hue)

        if valid_pixel_count < VALID_PIXEL_THRESHOLD:  # Minimum pixel threshold
            mean_scaled_hue = None
        else:
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

            print(f'Hue value of tube {tube}: {scaled_hue}')
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

        # Store the result with total pixels and valid pixel count
        results[tube] = {
            "hue": mean_scaled_hue,
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixel_count
        }

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
        # CROP_Y1, CROP_Y2, CROP_X1, CROP_X2 = 308, 418, 337, 887
        image = image[CROP_Y1:CROP_Y2, CROP_X1:CROP_X2]

        # Run the test tube detection algorithm
        results = detect_test_tube(image, output_dir=output_dir)

        # Save the image with annotations
        cv2.imwrite(output_image_path, image)
        print(f"Results: {results}")
        print(f"Annotated image saved at {output_image_path}")
    else:
        print("Failed to capture image.")
