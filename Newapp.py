# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf

from utils import load_data, num_to_char
from modelutil import load_model

# New imports for live webcam
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

# Set the layout to the streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipTranscribe')
    st.info('This application is originally developed from the LipNet deep learning model.')

# Title
st.title('LipReading Web-app')

# ----------------------------
# Section 1: Pre-recorded video demo (your existing code)
# ----------------------------

st.header(" Try with Sample Videos")
# Generate a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')
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

# ----------------------------
# Section 2: Live Webcam Lip Reading (NEW)
# ----------------------------

st.header("🎥 Live Webcam Lip Reading")

# Load your model once (reuse it in the video processor)
live_model = load_model()

# Output box for displaying transcription
output_text = st.empty()

# Define the live video processor class
class LiveLipReader(VideoProcessorBase):
    def __init__(self):
        self.frames_buffer = []
        self.frame_count = 75  # Expected sequence length for LipNet
        self.model = live_model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Here you need your face + mouth detection logic
        # For simplicity, we'll resize the full frame as a placeholder
        # Replace this with your `extract_mouth_region()` method
        processed_frame = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (100, 50)) / 255.0

        self.frames_buffer.append(processed_frame)

        # If we have enough frames, make a prediction
        if len(self.frames_buffer) == self.frame_count:
            # Convert to numpy array and expand dims
            sequence = np.array(self.frames_buffer)
            sequence = np.expand_dims(sequence, axis=-1)  # Add channel dim for grayscale
            sequence = np.expand_dims(sequence, axis=0)   # Batch dimension

            # Model prediction
            yhat = self.model.predict(sequence)
            decoder = tf.keras.backend.ctc_decode(yhat, [self.frame_count], greedy=True)[0][0].numpy()
            prediction_text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

            # Display prediction
            output_text.markdown(f"### 📝 Predicted Text: `{prediction_text}`")

            # Reset the buffer
            self.frames_buffer.clear()

        # Return the current video frame to display
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the webcam streamer
webrtc_streamer(
    key="live-lip-reader",
    video_processor_factory=LiveLipReader,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

