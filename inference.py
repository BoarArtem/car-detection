import cv2
import subprocess
from ultralytics import YOLO

model = YOLO("best.pt")


def get_stream_url(yt_url):
    result = subprocess.run(
        ["yt-dlp", "-g", yt_url],
        capture_output=True, text=True
    )
    return result.stdout.strip().split("\n")[0]

def realtime_inference(stream_url):
    cap = cv2.VideoCapture(stream_url)

    while True:
        res, frame = cap.read()

        result = model(frame, conf=0.4)
        annotated_frame = result[0].plot()

        cv2.imshow("Realtime car detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# yt_url = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
# print(f"Получена ссылка на трансляцию: {yt_url}")
# stream_url = get_stream_url(yt_url)
# print(f"Получен правильный формат ссылки для YOLO-модели: {stream_url}")
# realtime_inference(stream_url)