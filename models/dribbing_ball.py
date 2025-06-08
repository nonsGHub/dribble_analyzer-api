# dribbing_ball.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp
import io

# โหลดโมเดล
model_yolo = YOLO("C:/Users/1234n/basketballPJ/models/best.pt")
pose = mp.solutions.pose.Pose() # type: ignore

app = FastAPI()

# เปิด CORS ให้ ODC Studio เรียกใช้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน production ควรจำกัด domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_dribble_level(wrist_y, hip_y, knee_y, shoulder_y):
    if wrist_y > knee_y:
        return "Low"
    elif knee_y >= wrist_y > hip_y:
        return "Medium"
    else:
        return "High"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # วิเคราะห์ด้วย pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # วิเคราะห์ด้วย YOLO
    yolo_results = model_yolo(frame)

    # ดึงจุดจาก pose
    if not results.pose_landmarks:
        return {"error": "ไม่พบโครงกระดูกมนุษย์ในภาพ"}

    landmarks = results.pose_landmarks.landmark
    h, w, _ = frame.shape

    lwrist_y = landmarks[15].y * h
    rwrist_y = landmarks[16].y * h
    lhip_y = landmarks[23].y * h
    rhip_y = landmarks[24].y * h
    lknee_y = landmarks[25].y * h
    rknee_y = landmarks[26].y * h
    lshoulder_y = landmarks[11].y * h
    rshoulder_y = landmarks[12].y * h

    left_level = get_dribble_level(lwrist_y, lhip_y, lknee_y, lshoulder_y)
    right_level = get_dribble_level(rwrist_y, rhip_y, rknee_y, rshoulder_y)

    # ตรวจว่ามีลูกบาสใน YOLO หรือไม่
    ball_detected = False
    for box in yolo_results[0].boxes:
        cls_id = int(box.cls.cpu().numpy())
        name = model_yolo.names.get(cls_id, "")
        if name.lower() == "basketball":
            ball_detected = True
            break

    return {
        "ball_detected": ball_detected,
        "left_hand_dribble": left_level,
        "right_hand_dribble": right_level
    }
