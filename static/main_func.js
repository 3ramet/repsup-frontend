const urlParams = new URLSearchParams(window.location.search);
const id = urlParams.get("id");

fetch("/static/exercise_list.json")
    .then(res => res.json())
    .then(data =>{
        const item = data.find(d => d.id == id);
        if(item){
            document.getElementById("title").textContent = item.title
        }}
    )

function showPopup() {
  const popup = document.getElementById("popup-container");
  popup.classList.add("show"); // เปิดพร้อม animation
}

function hidePopup() {
  const popup = document.getElementById("popup-container");
  popup.classList.remove("show"); // ปิดพร้อม animation
  setTimeout(() => {
    popup.classList.add("popup-hidden"); // ซ่อนจริง ๆ หลัง animation จบ
  }, 300);
}

function showInfoPopup() {
  const popup = document.getElementById("popup-container-info");
  popup.classList.add("show"); // เปิดพร้อม animation
}

function hideInfoPopup() {
  const popup = document.getElementById("popup-container-info");
  popup.classList.remove("show"); // ปิดพร้อม animation
  setTimeout(() => {
    popup.classList.add("popup-hidden"); // ซ่อนจริง ๆ หลัง animation จบ
  }, 300);
}

window.addEventListener("message", function(event) {
  if (event.data.action === "yes") {
    window.location.href = "./";
    hidePopup();
    // document.getElementById("popup-container").style.display = "none";
  } else if (event.data.action === "no") {
    // alert("You chose NO → continue");
    hidePopup()
  }
});
