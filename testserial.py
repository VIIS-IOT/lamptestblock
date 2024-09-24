import serial
import time

# Setup the serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
time.sleep(2)  # Give time for the connection to establish

try:
    # Loop to accept commands and send them to Arduino
    while True:
        # Accept command from the user
        command = input("Enter command to send to Arduino (or 'exit' to quit): ").strip()
        
        if command.lower() == 'exit':
            print("Exiting...")
            break

        # Sending the entered command to Arduino
        ser.write(f'{command}\n'.encode('utf-8'))
        print(f"Sent: {command}")

        # Receiving data from Arduino
        time.sleep(0.1)  # Short delay to allow Arduino to respond
        while ser.in_waiting > 0:  # Check if there is data to read
            line = ser.readline().decode('utf-8').rstrip()
            if line:
                print(f"Received: {line}")

except KeyboardInterrupt:
    # If interrupted by user (Ctrl+C), close the serial connection
    print("Exiting...")

finally:
    # Make sure the serial connection is closed
    ser.close()