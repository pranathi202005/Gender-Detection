import streamlit as st
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model

# Load pre-trained gender classification model
model = load_model('./trainingDataTarget/model-019.model')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
labels_dict = {0: "Female", 1: "Male"}

def detect_gender_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    male_count = 0
    female_count = 0
    
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (32, 32))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 32, 32, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        if label == 1:
            male_count += 1
        else:
            female_count += 1
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(image, (x, y-40), (x+w, y), color_dict[label], -1)
        cv2.putText(image, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return image, male_count, female_count

def detect_gender_from_webcam():
    video = cv2.VideoCapture(0)
    male_count = 0
    female_count = 0
    
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (32, 32))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 32, 32, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            
            if label == 1:
                male_count += 1
            else:
                female_count += 1
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Webcam Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
    print(f"Males: {male_count}, Females: {female_count}")

# Streamlit app code
st.title("Gender Detection")

st.write("Upload an image or use the webcam for real-time gender detection")

# File uploader for user image input
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded image to OpenCV format
    image = np.array(Image.open(uploaded_file))

    # Detect gender on the uploaded image using the imported function
    output_image, male_count, female_count = detect_gender_from_image(image)

    # Convert OpenCV BGR format to RGB format for displaying in Streamlit
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the result
    st.image(output_image_rgb, caption="Processed Image with Gender Detection", use_column_width=True)
    st.write(f"Detected Males: {male_count}, Detected Females: {female_count}")

# Button to start webcam detection
if st.button("Start Webcam for Real-Time Detection"):
    detect_gender_from_webcam()
