import streamlit as st
import cv2
from inference import model, get_stream_url

if "running" not in st.session_state:
    st.session_state.running = False

st.title("Realtime car detection")

yt_url = st.text_input("Введите ссылку на трансляцию в ютубе")

confidence = st.slider(
    "Уверенеость модели",
    0.1,
    1.0,
    0.4,
    0.05
)

col1, col2 = st.columns(2)

with col1:
    if st.button("Запустить"):
        st.session_state.running = True
with col2:
    if st.button("Остановить"):
        st.session_state.running = False


frame_placeholder = st.empty()

if st.session_state.running:
    st.write("Получение потока...")
    stream_url = get_stream_url(yt_url)
    st.success(f"Поток подключен: {stream_url}")

    cap = cv2.VideoCapture(stream_url)

    while st.session_state.running:
        res, frame = cap.read()

        result = model(frame, conf=confidence)
        annotated_frame = result[0].plot()

        annotated_frame = cv2.cvtColor(
            annotated_frame,
            cv2.COLOR_BGR2RGB
        )
        frame_placeholder.image(
            annotated_frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()