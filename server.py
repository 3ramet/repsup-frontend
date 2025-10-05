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

@app.route("/mediapipe")
def mediapipe_page():
    return render_template("mediapipe.html")

@app.route("/pose_data", methods=["POST"])
def pose_data():
    raw_data = request.data.decode("utf-8")

    # 🔹 แยกส่วน id และข้อมูลจุด
    try:
        exercise_id, pose_str = raw_data.split("|", 1)
    except ValueError:
        return jsonify({"error": "invalid payload"}), 400

    # 🔹 แปลง string -> list ของ [x, y, z]
    pose_points = [
        [float(x), float(y), float(z)]
        for x, y, z in (p.split(",") for p in pose_str.split(";") if p)
    ]

    print(f"✅ ID={exercise_id}, ได้ {len(pose_points)} จุด")
    print("จุดแรก:", pose_points[0])

    # คุณสามารถทำการบันทึก / ประมวลผลต่อได้ที่นี่
    return jsonify({"bicep": True, "lateral": False, "squart": False})
    # return jsonify({"status": "received", "points": len(pose_points)})


    

if __name__ == "__main__":
    import eventlet
    eventlet.monkey_patch()
    # socketio.run(app, host="127.0.0.1", port=5000, debug=True)
    app.run(
        host="0.0.0.0",
        port=8080,
        ssl_context=("secrets/ssl.crt", "secrets/ssl_private.key"),
        debug=True)
