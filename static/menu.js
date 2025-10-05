let currentExerciseId = null; 

// Scrolling Function
document.addEventListener("DOMContentLoaded", () => {
  const welcome = document.querySelector(".welcome");
  const minHeight = 30; // ความสูงต่ำสุด

  window.addEventListener("scroll", () => {
    const newHeight = Math.max(window.innerHeight - window.scrollY, minHeight);
    welcome.style.height = newHeight + "px";
  });
});

// load json
document.addEventListener("DOMContentLoaded", () => {

  fetch("/static/exercise_list.json")
    .then(res => res.json())
    .then(data =>{
      const container = document.getElementById("card-container");

      data.forEach(item => {
          const card = document.createElement("a");
          // card.href = `./main_func.html?id=${item.id}`;
          // card.href = `/main_func?id=${item.id}`;
          card.className = "card";
          card.addEventListener("click", function(event) {
            showPopup();

            document.getElementById("popup_title").textContent = item.title;
            document.getElementById("popup-detail").textContent = item.detail;
            document.getElementById("popup-notice").textContent = item.notice;

            // ใช้ ex_pic จาก console.log
            document.getElementById("popup-pic").src = item.ex_pic || item.pic;
            // window.location.href = `pop-up.html?id=${item.id}`;
            currentExerciseId = item.id; 
          });
          card.innerHTML = `
            <img src="./static/${item.pic}" alt="รูปภาพตัวอย่าง">
              <div class="card-content">
                <div class="card-content-head">
                  <h3 class="font-bold text-gray-800">${item.title}</h3>
                  <p>ระดับ : ${item.lv}</p>
                </div>
                <div class="card-content-tag">
                  <p>${item.body}</p>
                  <p>${item.type}</p>
                </div>
                <ul class="detail text-gray-600 font-light">
                  <li>${item.desc}</li>
                </ul>
              </div>
          `;
          container.appendChild(card)
      });
    })
  });

function showPopup() {
  const popup = document.getElementById("popup-container");
  document.body.classList.add('no-scroll');
  popup.classList.add("show"); // เปิดพร้อม animation
}

function hidePopup() {
  const popup = document.getElementById("popup-container");
  popup.classList.remove("show"); // ปิดพร้อม animation
  document.body.classList.remove('no-scroll');
  setTimeout(() => {
    popup.classList.add("popup-hidden"); // ซ่อนจริง ๆ หลัง animation จบ
  }, 300);
}

function startProgram() {
    if (currentExerciseId) {
        // เปลี่ยนหน้าไปยัง URL ที่ต้องการพร้อมส่งค่า id ไปด้วย
        // console.log(currentExerciseId)
        window.location.href = `/main_func?id=${currentExerciseId}`;
    } else {
        hidePopup();
    }
}