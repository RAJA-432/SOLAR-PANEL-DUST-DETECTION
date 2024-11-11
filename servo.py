import tensorflow as tf
import cv2
import numpy as np
import pyfirmata
from pyfirmata import Arduino, util
import time

# Load the pre-trained model
model_path = "C:\design project\Models\Teja.h5"
model = tf.keras.models.load_model(model_path)

# Set the path to the new image to be classified
image_path = "C:\design project\download (1).jpeg"

# Load and preprocess the image
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = np.array(img)
img = tf.keras.applications.mobilenet.preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Use the pre-trained model to predict the class of the image
predictions = model.predict(img)

# Establish communication with the Arduino board
board = pyfirmata.Arduino('COM7')  # Replace '/dev/ttyACM0' with the appropriate port for your system
it = util.Iterator(board)
it.start()

# Get the servo motor pin
servo_pin = board.get_pin('d:9:s')  # Replace '9' with the digital pin number you connected the servo to

if predictions[0][0] > 0.5:  # Dusty panel
    print("Dusty")
    start_time = time.time()
    while time.time() - start_time < 30:  # Run for 30 seconds
        servo_pin.write(180)  # Move to 180 degrees
        time.sleep(1)  # Wait for 1 second
        servo_pin.write(0)    # Move back to 0 degrees
        time.sleep(1)  # Wait for 1 second
else:  # Clean panel
    print("Clean")
    servo_pin.write(0)  # Move the servo to the initial position (0 degrees)

# Close the connection to the Arduino board
board.exit()