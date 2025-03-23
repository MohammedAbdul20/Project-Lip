# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import cv2
import numpy as np
import tensorflow as tf 

from utils import num_to_char
from modelutil import load_model
from collections import deque

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipTranscribe')
    st.info('This application is originally developed from the LipNet deep learning model.')
    
    # Button to activate live webcam lipreading
    use_webcam = st.checkbox('Use Live Webcam')

st.title('LipReading Web-app')

# ========== If user selects Webcam ==========
if use_webcam:
    st.header('Live Webcam LipReading with Sliding Window')
    
    # Load your trained model
    model = load_model()

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        # Create Streamlit placeholders
        frame_placeholder = st.empty()
        prediction_placeholder = st.empty()

        st.info("Press STOP in the toolbar to exit live webcam lipreading.")
        
        # Define a sliding window / buffer for frames
        frame_buffer = deque(maxlen=75)  # 75 frames expected by model

        # Streamlit loop
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame.")
                break

            # Preprocess frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped_frame = gray_frame[190:236, 80:220]  # Crop lip region

            # Normalize the frame
            input_frame = cropped_frame.astype(np.float32)
            mean = np.mean(input_frame)
            std = np.std(input_frame)
            normalized_frame = (input_frame - mean) / (std + 1e-10)

            # Add a channel dimension (H, W, 1)
            normalized_frame = np.expand_dims(normalized_frame, axis=-1)

            # Append to sliding window buffer
            frame_buffer.append(normalized_frame)

            # Display live webcam feed
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_bgr, channels='RGB', caption='Webcam Feed')

            # If we have 75 frames, make prediction
            if len(frame_buffer) == 75:
                # Convert buffer to numpy array and add batch dim
                input_frames = np.array(frame_buffer)
                input_frames = np.expand_dims(input_frames, axis=0)  # Shape: (1, 75, 46, 140, 1)

                # Predict
                yhat = model.predict(input_frames)
                decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

                # Convert prediction to text
                converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

                # Display prediction
                prediction_placeholder.subheader(f'Prediction: {converted_prediction}')

        cap.release()

# ========== If user selects pre-recorded video ==========
else:
    # Generating a list of options or videos 
    options = os.listdir(os.path.join('..', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)

    # Generate two columns 
    col1, col2 = st.columns(2)

    if options: 
        # Rendering the video 
        with col1: 
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('..','data','s1', selected_video)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

            # Rendering inside of the app
            video = open('test_video.mp4', 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)

        with col2: 
            st.info('This is all the machine learning model sees when making a prediction')
            from utils import load_data  # Moved import to avoid conflict above
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
