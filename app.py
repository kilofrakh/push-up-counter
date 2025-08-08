import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import mediapipe as mp
import numpy as np
import math
import queue
import time
import cv2

class PoseDetector():
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackingCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if draw and self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(
                img, 
                self.results.pose_landmarks, 
                self.mpPose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)
        return self.results
    
    def findPosition(self, img):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
        return self.lmList
    
    def findAngle(self, p1, p2, p3):
        if len(self.lmList) == 0:
            return 0
            
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]
        
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        if angle > 180:
            angle = 360 - angle
        elif angle < 0:
            angle = -angle
            
        return angle

class PushUpProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = PoseDetector(detectionCon=0.8)
        self.count = 0.0
        self.dir = 0
        self.per = 0
        self.pTime = time.time()
        self.frame_count = 0
        self.fps = 0
        self.result_queue = queue.Queue()
        self.status = "Waiting for detection..."
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))
        self.frame_count += 1
        
        if self.frame_count % 2 == 0:
            results = self.detector.findPose(img)  
            lmList = self.detector.findPosition(img)
            
            if len(lmList) > 0:
                if (len(lmList) > 31 and len(lmList) > 29 and 
                    lmList[31][2] + 50 > lmList[29][2] and 
                    lmList[32][2] + 50 > lmList[30][2]):
                    
                    angle = self.detector.findAngle(11, 13, 15)
                    self.per = -1.25 * angle + 212.5
                    self.per = max(0, min(100, self.per))
                    
                    if self.per >= 95:
                        if self.dir == 0:
                            self.count += 0.5
                            self.dir = 1
                            self.result_queue.put((self.count, self.per, "Down position"))
                    elif self.per <= 5:
                        if self.dir == 1:
                            self.count += 0.5
                            self.dir = 0
                            self.result_queue.put((self.count, self.per, "Up position"))
                    
                    self.status = "Tracking push-ups..."
                else:
                    self.status = "Take your position..."
            else:
                self.status = "No person detected..."
            
            cTime = time.time()
            self.fps = 1 / (cTime - self.pTime)
            self.pTime = cTime
            
            self.result_queue.put((self.count, self.per, self.status, self.fps))
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.set_page_config(page_title="Push-Up Counter", layout="centered")
st.title("Real-Time Push-Up Counter")

col1, col2 = st.columns(2)
with col1:
    count_placeholder = st.empty()
    count_placeholder.metric("Push-Up Count", "0.0")
with col2:
    form_placeholder = st.empty()
    form_placeholder.metric("Form Completion", "0%")

status_placeholder = st.empty()
fps_placeholder = st.empty()
progress_placeholder = st.empty()

if st.button("Reset Counter"):
    if 'webrtc_ctx' in st.session_state:
        st.session_state.webrtc_ctx.video_processor.count = 0.0
        st.session_state.webrtc_ctx.video_processor.dir = 0
        st.session_state.webrtc_ctx.video_processor.result_queue.queue.clear()
    st.rerun()

webrtc_ctx = webrtc_streamer(
    key="pushup-counter",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=PushUpProcessor,
    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480,
            "frameRate": 30
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

if webrtc_ctx and webrtc_ctx.video_processor:
    while True:
        try:
            count, per, status, fps = webrtc_ctx.video_processor.result_queue.get(timeout=1.0)
            count_placeholder.metric("Push-Up Count", f"{count:.1f}")
            form_placeholder.metric("Form Completion", f"{int(per)}%")
            status_placeholder.info(f"Status: {status}")
            fps_placeholder.caption(f"FPS: {int(fps)}")
            progress_placeholder.progress(int(per)/100)
        except queue.Empty:
            pass
        except ValueError:
           
            pass
