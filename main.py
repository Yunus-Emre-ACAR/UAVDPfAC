import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Live UAV detection with YOLOv8 + OpenCV")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Path to a YOLOv8 .pt model (ideally trained on drone/UAV data)")
    parser.add_argument("--source", type=str, default="0",
                         help="Camera index (e.g. 0), video file path, or stream URL")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default=None,
                         help="Force device, e.g. 'cpu' or '0' for first GPU (default: auto)")
    parser.add_argument("--save", type=str, default=None,
                         help="Optional path to save annotated output video (e.g. out.mp4)")
    return parser.parse_args()


def open_source(source: str):
    # Camera indices come in as strings from argparse; convert if numeric
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()

    model = YOLO(args.model)
    cap = open_source(args.source)

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

    prev_time = time.time()
    fps_display = 0.0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or camera read failed.")
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        result = results[0]

        annotated = result.plot()  # frame with boxes/labels drawn

        # FPS overlay
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)
        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        num_detections = len(result.boxes) if result.boxes is not None else 0
        cv2.putText(annotated, f"Detections: {num_detections}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("UAV Detector", annotated)

        if writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()