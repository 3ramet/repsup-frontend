import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10";

const video = document.getElementById("video");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

let poseLandmarker;
let running = false;

// ✅ เริ่มต้นระบบ
async function initPose() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  startCamera();
}

// ✅ เปิดกล้อง
async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    running = true;
    detectPose();
  };
}

async function detectPose() {
  if (!running || !poseLandmarker) return;

  const results = await poseLandmarker.detectForVideo(video, performance.now());

  // วาดภาพกล้อง
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // วาดโครงร่างร่างกาย
  if (results.landmarks) {
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmarks of results.landmarks) {
      drawingUtils.drawLandmarks(landmarks, { color: "red", radius: 3 });
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: "lime",
        lineWidth: 2,
      });
    }
  }

  requestAnimationFrame(detectPose); // เรียกซ้ำทุก frame
}


initPose();
