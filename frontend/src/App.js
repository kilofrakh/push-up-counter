import React, { useEffect, useRef, useState } from 'react';
import './App.css';

import { Pose } from '@mediapipe/pose';
import { drawConnectors, drawLandmarks } from '@mediapipe/drawing_utils';
import { Camera } from '@mediapipe/camera_utils';

// Utility to compute angle between three points p1->p2->p3 in degrees
function computeAngleDegrees(p1, p2, p3) {
  const radians = Math.atan2(p3.y - p2.y, p3.x - p2.x) - Math.atan2(p1.y - p2.y, p1.x - p2.x);
  let angle = (radians * 180) / Math.PI;
  if (angle > 180) angle = 360 - angle;
  else if (angle < 0) angle = -angle;
  return angle;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function CameraPushupCounter() {
  const VIDEO_WIDTH = 480;
  const VIDEO_HEIGHT = 360;
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [count, setCount] = useState(0);
  const dirRef = useRef(0); // 0 going down, 1 going up (mirrors python logic)
  const lastPerRef = useRef(0);
  const cameraRef = useRef(null);
  const [cameraError, setCameraError] = useState(null);

  useEffect(() => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext('2d');

    // Use CDN path for wasm and assets
    const pose = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    pose.setOptions({
      modelComplexity: 0, // <---- reduced for speed
      smoothLandmarks: true,
      enableSegmentation: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    pose.onResults((results) => {
      // Draw the camera frame
      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

      if (results.poseLandmarks && results.poseLandmarks.length) {
        // Draw pose
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#00FF00', lineWidth: 2 });
        drawConnectors(canvasCtx, results.poseLandmarks, Pose.POSE_CONNECTIONS, { color: '#FF7F40', lineWidth: 2 });

        // Use left arm: indices 11 (shoulder), 13 (elbow), 15 (wrist)
        const lm = results.poseLandmarks;
        const getPoint = (idx) => ({ x: lm[idx].x * canvasElement.width, y: lm[idx].y * canvasElement.height });
        const p1 = getPoint(11);
        const p2 = getPoint(13);
        const p3 = getPoint(15);

        const angle = computeAngleDegrees(p1, p2, p3);
        // Map angle to percentage per your python logic: per = -1.25 * angle + 212.5 then clamp 0..100
        let per = -1.25 * angle + 212.5;
        per = clamp(per, 0, 100);
        lastPerRef.current = per;

        // Bar visualization
        const bar = ((VIDEO_HEIGHT - 60) * (per / 100)) + 60; // inverse mapping like np.interp to [100..650]
        // Draw percentage text and bar
        canvasCtx.fillStyle = per >= 95 || per <= 5 ? '#00FF00' : '#FF0000';
        canvasCtx.font = '24px Arial';
        canvasCtx.fillText(`${Math.round(per)}%`, VIDEO_WIDTH - 100, 40);
        canvasCtx.strokeStyle = '#FF0000';
        canvasCtx.lineWidth = 3;
        // Bar background
        canvasCtx.strokeRect(VIDEO_WIDTH - 75, 60, 20, VIDEO_HEIGHT - 100);
        // Bar fill
        canvasCtx.fillStyle = '#FF0000';
        const barTop = Math.max(60, Math.min(VIDEO_HEIGHT - 40, bar));
        canvasCtx.fillRect(VIDEO_WIDTH - 75, barTop, 20, (VIDEO_HEIGHT - 40) - barTop);

        // Count logic (mirrors python):
        // if per >= 95 and dir == 0 => count += 0.5, dir = 1
        // elif per <= 5 and dir == 1 => count += 0.5, dir = 0
        if (per >= 95) {
          if (dirRef.current === 0) {
            setCount((c) => c + 0.5);
            dirRef.current = 1;
          }
        } else if (per <= 5) {
          if (dirRef.current === 1) {
            setCount((c) => c + 0.5);
            dirRef.current = 0;
          }
        }

        // Count box
        // canvasCtx.fillStyle = 'rgba(0,255,0,0.7)';
        // canvasCtx.fillRect(12, 6, 200, 45);
        // canvasCtx.fillStyle = '#FF0000';
        // canvasCtx.font = '32px Arial';
        // canvasCtx.fillText(`count: ${Math.floor(count)}`, 20, 38);
      }

      canvasCtx.restore();
    });

    // Prepare camera
    const startCamera = async () => {
      canvasElement.width = VIDEO_WIDTH;
      canvasElement.height = VIDEO_HEIGHT;
      try {
        cameraRef.current = new Camera(videoElement, {
          onFrame: async () => {
            await pose.send({ image: videoElement });
          },
          width: VIDEO_WIDTH,
          height: VIDEO_HEIGHT,
        });
        await cameraRef.current.start();
        setCameraError(null);
      } catch (e) {
        setCameraError('Unable to access camera. Please allow camera permission or check device.');
      }
    };

    startCamera();

    return () => {
      if (cameraRef.current) {
        try { cameraRef.current.stop(); } catch (_) {}
      }
    };
  }, []);

  return (
    <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 20}}>
      <h1>Live Push-Up Counter</h1>
      {cameraError && (
        <div style={{color: 'red', fontWeight: 'bold', marginBottom: 16}}>{cameraError}</div>
      )}
      <video ref={videoRef} style={{ display: 'none' }}></video>
      <canvas ref={canvasRef} style={{ borderRadius: 8, border: '2px solid #1976d2' }} />
      {/* Count is now only displayed in web div below! */}
      <div style={{marginTop: 12, fontSize: 28, fontWeight: 'bold'}}>Count: {Math.floor(count)}</div>
    </div>
  );
}

function App() {
  return (
    <CameraPushupCounter />
  );
}

export default App;
