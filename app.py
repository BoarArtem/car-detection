import streamlit as st
import cv2
from inference import model, get_stream_url

if "running" not in st.session_state:
    st.session_state.running = False

st.title("Realtime car detection")

yt_url = st.text_input("Enter the link to the YouTube broadcast")

confidence = st.slider(
    "Model confidence",
    0.1,
    1.0,
    0.4,
    0.05
)

col1, col2 = st.columns(2)

with col1:      
    if st.button("Start"):
        st.session_state.running = True
with col2:
    if st.button("Stop"):
        st.session_state.running = False


frame_placeholder = st.empty()

if st.session_state.running:
    st.write("Receiving stream...")
    stream_url = get_stream_url(yt_url)
    st.success(f"Stream connected: {stream_url}")

    cap = cv2.VideoCapture(stream_url)

    while st.session_state.running:
        res, frame = cap.read()

        result = model(frame, conf=confidence)
        annotated_frame = result[0].plot()

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )

        cars_number = len(result[0].boxes)

        cv2.putText(annotated_frame, f"Cars: {cars_number}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_placeholder.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()
