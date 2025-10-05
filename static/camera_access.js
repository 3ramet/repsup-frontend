import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.10";

const camera = document.getElementById("camera");
const canvas = document.getElementById("output");

const count_txt = document.getElementById("count_txt");

const ctx = canvas.getContext("2d");

const urlParams = new URLSearchParams(window.location.search);
const exerciseId  = urlParams.get("id");

let poseLandmarker;
let running = false;

let isReady = false;
let onStart = false;

let isRunning = false;

let intervalId = null; // เก็บ ID ของ setInterval
// const socket = io("http://127.0.0.1:5000");

// socket.on("connect", () => console.log("Socket.IO connected"));
// socket.on("response", (data) => console.log("Server response:", data));

document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM content loaded.");
  // showCamera();
  // camera.style.display = "flex";
  initPose();
});

async function showCamera() {
  await navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      camera.srcObject = stream;
      camera.onloadedmetadata = () => {
        camera.play();
        canvas.width = camera.videoWidth;
        canvas.height = camera.videoHeight;
        running = true;
      };
    })
    .catch(error => {
      console.error("Cannot access camera", error);
    });
    
}

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

  showCamera();
}

let position = "down";
let pose_count = 0;

async function detectPose() {
  if (!onStart || !poseLandmarker) return;

  const results = await poseLandmarker.detectForVideo(camera, performance.now());
  
  // วาดภาพกล้อง
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(camera, 0, 0, canvas.width, canvas.height);
  
  if (results.landmarks && results.landmarks.length > 0) {
    const l = results.landmarks[0];
    const leftShoulder = l[11], rightShoulder = l[12];
    const leftWrist  = l[15], rightWrist = l[16];
    const leftElbow = l[13], rightElbow = l[14];
    const leftHip = l[23], rightHip = l[24];

    const midX = (leftShoulder.x + rightShoulder.x + leftHip.x + rightHip.x) / 4;
    const midY = (leftShoulder.y + rightShoulder.y + leftHip.y + rightHip.y) / 4;

    window.isReady = midX > 0.2 && midX < 0.8 && midY > 0.2 && midY < 0.8;
    
    if (window.isReady){
      sendFrame(results.landmarks[0]);
      // console.log("frame send");
      if (exerciseId == 1) { // 1 = bicep curl
        // ตรวจว่า landmark มีค่าครบไหม
        if (rightWrist && rightElbow && leftWrist && leftElbow) {
          // ยกขึ้น: wrist สูงกว่า elbow (y น้อยกว่า)
          if (rightWrist.y < rightElbow.y - 0.1 && leftWrist.y < leftElbow.y - 0.1 && position === "down") {
            position = "up";
            pose_count += 1;
            updateScore(pose_count);
            // console.log("Count:", pose_count);
          }

          if (rightWrist.y > rightElbow.y + 0.1 && leftWrist.y > leftElbow.y + 0.1) {
            position = "down";
          }
        }
      }
      else if(exerciseId == 2){
        if (rightWrist && rightShoulder && leftWrist && leftShoulder) {
          // ยกขึ้น: wrist สูงกว่า elbow (y น้อยกว่า)
          if (rightWrist.y < rightShoulder.y - 0.05 && leftWrist.y < leftShoulder.y - 0.1 && position === "down") {
            position = "up";
            pose_count += 1;
            updateScore(pose_count);
            // console.log("Count:", pose_count);
          }

          if (rightWrist.y > rightShoulder.y + 0.05 && leftWrist.y > leftShoulder.y + 0.1) {
            position = "down";
          }
        }
      }
    }
 

    
  }

  if (onStart) requestAnimationFrame(detectPose);
}

// ส่ง frame ไป server
async function sendFrame(landmarks) {
  if (!landmarks || landmarks.length === 0) return;

  try {
    // เตรียมข้อมูลแต่ละจุดเป็น array ของ {x, y, z, visibility}
    const poseString = landmarks
      .map(l => `${l.x.toFixed(4)},${l.y.toFixed(4)},${l.z.toFixed(4)}`)
      .join(";");

    const payload = `${exerciseId}|${poseString}`;
    // console.log(payload);
    // ส่งข้อมูลแบบ JSON ด้วย HTTPS
    const response = await fetch("https://172.24.5.192:8080/pose_data", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: payload
    });

    if (!response.ok) {
      console.error("❌ Server error:", response.statusText);
    } else {
      const data = await response.json();
      // console.log("✅ Server response:", data);
    }
  } catch (err) {
    console.error("⚠️ Error sending pose data:", err);
  }
}

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const overlay = document.getElementById("overlay");

startBtn.addEventListener("click", () => {
  startBtn.style.display = "none";
  stopBtn.style.display = "flex";
  canvas.style.display = "block";
  camera.style.display ="none";
  onStart = true;
  pose_count = 0;
  // เริ่มตรวจสอบตำแหน่งผู้ใช้แทน countdown
  checkUserPositionLoop();

  detectPose();

});

function checkUserPositionLoop() {
  // loop ตรวจสอบทุก 100 ms
  const checkInterval = setInterval(() => {
    if (!onStart) {
      // หาก Stop ให้หยุด loop
      clearInterval(checkInterval);
      overlay.style.display = "none";
      return;
    }

    if (window.isReady) {
      // ผู้ใช้ยืนถูกกรอบ
      overlay.style.display = "none";
      stopBtn.style.display = "flex";

      if (!intervalId) {
        intervalId = setInterval(sendFrame, 1000);
        isRunning = true;
        console.log("✅ Started sending frames");
      }
    } else {
      // ผู้ใช้ยังไม่อยู่ในกรอบ
      overlay.style.display = "flex";
    }
  }, 100);
}
// document.addEventListener('click', function(event) {
//   updateScore(1);
// });

function updateScore(score){
  // count_txt.classList.remove('animate__bounceIn');
  count_txt.textContent = score;
  count_txt.classList.add('animate__animated', 'animate__bounceIn')
  count_txt.addEventListener('animationend', () => {
    count_txt.classList.remove('animate__animated', 'animate__bounceIn');
  }, { once: true });

}

// กด Stop → หยุดส่ง frame
stopBtn.addEventListener("click", () => {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
    console.log("🛑 Stopped sending frames");
  }
  isRunning = false;
  onStart = false;
  canvas.style.display = "none";
  camera.style.display ="block";
  startBtn.style.display = "flex";
  stopBtn.style.display = "none";
  stopBtn.classList.remove("glowing")
});

