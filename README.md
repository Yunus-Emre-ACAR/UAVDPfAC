# UAVDPfAC
UAV detection program for aviation club

UAV Camera Detector
--------------------
Opens a camera (webcam or video file/stream) and runs a YOLOv8 model
to detect UAVs/drones in each frame, drawing bounding boxes live.

Requirements:
    pip install ultralytics opencv-python --break-system-packages

Usage:
    python main.py --model best.pt --source 0
    python main.py --model best.pt --source video.mp4
    python main.py --model best.pt --source rtsp://...

Notes:
    - "best.pt" should be a YOLOv8 model trained/fine-tuned on drone data.
      If you don't have one yet, you can test the pipeline with the stock
      "yolov8n.pt" (won't have a "drone" class, but proves the camera loop
      and inference pipeline work end-to-end).
    - --imgsz can be bumped up (e.g. 960 or 1280) to help with small,
      distant UAV targets against sky background, at some FPS cost.
    - --conf controls the confidence threshold; lower it if you're missing
      small/faint detections, raise it if you get too many false positives.
