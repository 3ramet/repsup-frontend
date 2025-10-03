from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import io

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")  # เปิดไฟล์ HTML จริง

@app.route("/main_func")
def main_func():
    # อ่านค่า id จาก query string
    card_id = request.args.get('id')
    return render_template("main_func.html", card_id=card_id)

@app.route("/popup")
def popup():
    return render_template("pop-up.html")

@app.route("/popup-info")
def popup_info():
    return render_template("pop-up-info.html")

@socketio.on("frame")
def handle_frame(data):
    """
    data = ArrayBuffer (binary) จาก client
    """
    # แปลง bytes → Image
    try:
         # 1️⃣ ตรวจสอบ type ของข้อมูล
        print("Received type:", type(data))
        if isinstance(data, bytes):
            print("Received bytes length:", len(data))
        else:
            print("Received non-bytes, type:", type(data))

        # 2️⃣ แปลงเป็น Image (ถ้าเป็น bytes)
        if isinstance(data, bytes):
            img = Image.open(io.BytesIO(data))
            print("Image size:", img.size)
            # สามารถบันทึกเป็นไฟล์ทดสอบได้
            img.save("test_frame.jpg")
        emit("response", {"status": "ok"})
    except Exception as e:
        print("Error processing frame:", e)
        emit("response", {"status": "error", "msg": str(e)})


    

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    # socketio.run(app, host="127.0.0.1", port=5000, debug=True)
    app.run(
        host="0.0.0.0",
        port=8080,
        # ssl_context=("ssl.crt", "ssl_private.key"),
        debug=True)
