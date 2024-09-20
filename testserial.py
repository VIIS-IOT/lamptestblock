import serial
import time

# Setup the serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Give time for the connection to establish

try:
    # Loop to send 'get_data' every 1 second
    while True:
        # Sending 'get_data' to Arduino
        ser.write(b'get_data\n')
        print("Sent: get_data")

        # Receiving data from Arduino
        line = ser.readline().decode('utf-8').rstrip()
        if line:
            print(f"Received: {line}")
        
        # Wait for 1 second before sending the next request
        time.sleep(1)

except KeyboardInterrupt:
    # If interrupted by user (Ctrl+C), close the serial connection
    print("Exiting...")

finally:
    # Make sure the serial connection is closed
    ser.close()

