# Organic-and-Recyclable-west-segregation
Trash Classification with Camera and Arduino
============================================

This Python script captures video from the camera, classifies trash as "Organic" or "Recyclable" using a trained deep learning model, 
and sends signals to an Arduino for LED control.

----------------------------------------------------------------------------------

# 🚀 IMPORT REQUIRED LIBRARIES
import cv2  # OpenCV for camera access and image processing
import numpy as np  # NumPy for handling image arrays
import tensorflow as tf  # TensorFlow for loading and running the model
import time  # Used to slow down frame processing
import serial  # Serial communication with Arduino

# ----------------------------------------------------------------------------------
# 🔹 INITIALIZE ARDUINO COMMUNICATION
# ----------------------------------------------------------------------------------
try:
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)
except serial.SerialException:
    print("⚠️ Arduino not detected on COM4. Check connection!")

# ----------------------------------------------------------------------------------
# 🧠 LOAD PRE-TRAINED WASTE CLASSIFICATION MODEL
# ----------------------------------------------------------------------------------
MODEL_PATH = "C:/Users/balaj/tf_env/mi_pro/recyclable_vs_organic.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------------------------------------------------------------
# 🖼️ FUNCTION: PREPROCESS IMAGE
# ----------------------------------------------------------------------------------
def preprocess_image(image):
    """
    Converts an image from BGR to RGB, resizes it to (128,128),
    normalizes pixel values, and expands dimensions to match model input shape.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    resized = cv2.resize(image, (128, 128))  # Resize to match training data
    normalized = resized / 255.0  # Normalize pixel values to range [0, 1]
    return np.expand_dims(normalized, axis=0)  # Add batch dimension (1, 128, 128, 3)

# ----------------------------------------------------------------------------------
# 🗑️ FUNCTION: CLASSIFY WASTE TYPE
# ----------------------------------------------------------------------------------
def classify_trash(image):
    """
    Predicts whether an image is organic or recyclable.
    Sends a signal to the Arduino for LED control.
    """
    processed = preprocess_image(image)  # Preprocess image
    prediction = model.predict(processed)[0][0]  # Get model confidence score

    # Print confidence score for debugging (optional)
    print(f"Confidence: {prediction:.2f}")

    if prediction < 0.5:
        arduino.write(b'1')  # Send '1' to Arduino for organic waste
        return "Organic"
    else:
        arduino.write(b'0')  # Send '0' to Arduino for recyclable waste
        return "Recyclable"

# ----------------------------------------------------------------------------------
# 📷 FUNCTION: START CAMERA & REAL-TIME CLASSIFICATION
# ----------------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)  # Open default camera
    frame_count = 0  # Counter to process every 5th frame

    while True:
        ret, frame = cap.read()  # Capture frame from camera
        if not ret:
            break  # Exit if frame not captured correctly

        frame_count += 1

        # Process every 5th frame for better accuracy & performance
        if frame_count % 5 == 0:
            label = classify_trash(frame)  # Get classification result

            # Display classification result on the video frame
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Trash Classifier', frame)  # Show video feed with predictions

        time.sleep(0.2)  # Slow down processing for stability

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------
# 🏁 RUN THE PROGRAM
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

----------------------------------------------------------------------------------
COMPONENT EXPLANATION:
----------------------------------------------------------------------------------

1️⃣ **Import Libraries**  
- `cv2` (OpenCV): Accesses the camera and processes images.  
- `numpy`: Handles image arrays and numerical operations.  
- `tensorflow`: Loads and runs the trained deep learning model.  
- `time`: Introduces small delays to optimize processing.  
- `serial`: Enables communication with the Arduino to control LEDs.

2️⃣ **Initialize Arduino**  
- Establishes serial communication with the Arduino on COM4.  
- If Arduino is not connected, an error message is printed.

3️⃣ **Load Model**  
- Loads the pre-trained deep learning model for trash classification.

4️⃣ **Preprocess Image**  
- Converts image from BGR to RGB (since OpenCV reads images in BGR format).  
- Resizes it to (128, 128) to match the model's training size.  
- Normalizes pixel values between 0 and 1.  
- Expands dimensions to match the input shape required by the model.

5️⃣ **Classify Trash**  
- Runs the model on the preprocessed image.  
- If confidence score < 0.5, it classifies the image as "Organic" and sends a signal (`b'1'`) to the Arduino.  
- If confidence score ≥ 0.5, it classifies the image as "Recyclable" and sends a signal (`b'0'`) to the Arduino.  

6️⃣ **Main Function (Camera & Real-Time Processing)**  
- Captures frames from the camera.  
- Processes every 5th frame to improve performance.  
- Displays classification results on the video feed.  
- Sends classification results to the Arduino.  
- Stops when the user presses 'q'.  

7️⃣ **Run the Program**  
- Executes the `main()` function if the script is run directly.

----------------------------------------------------------------------------------
