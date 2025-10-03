const camera = document.getElementById("camera");

let isRunning = false;

let intervalId = null; // เก็บ ID ของ setInterval
const socket = io("http://127.0.0.1:5000");

socket.on("connect", () => console.log("Socket.IO connected"));
socket.on("response", (data) => console.log("Server response:", data));

document.addEventListener('DOMContentLoaded', function() {
  console.log("DOM content loaded.");
  showCamera();
});

async function showCamera() {
  await navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      camera.srcObject = stream;
    })
    .catch(error => {
      console.error("Cannot access camera", error);
    });
}

// ส่ง frame ไป server
function sendFrame() {
  if (camera.videoWidth === 0 || camera.videoHeight === 0) return;

  const canvas = document.createElement("canvas");
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(camera, 0, 0);

  canvas.toBlob((blob) => {
    const reader = new FileReader();
    reader.onload = () => {
      socket.emit("frame", reader.result); // ส่ง ArrayBuffer ไป server
    };
    reader.readAsArrayBuffer(blob);
  }, "image/jpeg", 0.5);
}

// กด Start → เริ่มส่ง frame ทุก 1 วินาที
document.getElementById("startBtn").addEventListener("click", () => {
  const overlay = document.getElementById("overlay");

  overlay.style.display = "flex";
  setTimeout(() => {
    overlay.style.display = "none";
    stopBtn.style.display = "inline-block";
    stopBtn.classList.add("glowing");
    if (!intervalId) {
      intervalId = setInterval(sendFrame, 1000);
      document.getElementById("startBtn").style.display = "none";
      document.getElementById("stopBtn").style.display = "flex";
      isRunning = true;
      console.log("✅ Started sending frames");
      stopBtn.classList.add("glowing")
    }
  }, 3000);
});

// กด Stop → หยุดส่ง frame
document.getElementById("stopBtn").addEventListener("click", () => {
  if (intervalId) {
    clearInterval(intervalId);
    intervalId = null;
    isRunning = false;
    document.getElementById("startBtn").style.display = "flex";
    document.getElementById("stopBtn").style.display = "none";
    stopBtn.classList.remove("glowing")
    console.log("🛑 Stopped sending frames");
  }
});

// function tryExit() {
//       // if (isRunning) {
//         // ถ้ามีฟังก์ชันทำงานอยู่ แสดง pop-up
//         document.getElementById("exitModal").style.display = "flex";
//       // } else {
//       //   // ถ้าไม่มีฟังก์ชันทำงาน ออกไปได้เลย
//       //   window.location.href = "./";
//       // }
// }

//     // ปิด pop-up
//     function closeExitPopup() {
//       document.getElementById("exitModal").style.display = "none";
//     }

//     // เมื่อกดยืนยันออก
//     function confirmExit() {
//       if (intervalId) {
//         clearInterval(intervalId);
//         intervalId = null;
//         isRunning = false;
//         stopBtn.classList.remove("glowing")
//         console.log("🛑 Stopped sending frames");
//         window.location.href = "./"; // redirect
//       }
//     }
