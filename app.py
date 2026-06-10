import streamlit as st
import cv2

from inference import get_stream_url, model

st.set_page_config(layout="wide")

st.title("Realtime Car Detection")

# session state
if "running" not in st.session_state:
    st.session_state.running = False

yt_url = st.text_input(
    "Введите ссылку на YouTube трансляцию"
)

confidence = st.slider(
    "Уверенность модели",
    0.1,
    1.0,
    0.4,
    0.05
)

col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Запустить"):
        st.session_state.running = True

with col2:
    if st.button("⏹ Остановить"):
        st.session_state.running = False

frame_placeholder = st.empty()

if st.session_state.running:

    st.write("Получение потока...")

    stream_url = get_stream_url(yt_url)

    st.success("Поток подключен")

    cap = cv2.VideoCapture(stream_url)

    while st.session_state.running:

        success, frame = cap.read()

        if not success:
            st.error("Ошибка чтения потока")
            break

        results = model(frame, conf=confidence)

        annotated_frame = results[0].plot()

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