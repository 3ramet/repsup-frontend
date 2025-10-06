import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify  # noqa: E402
from flask_socketio import SocketIO  # noqa: E402

from ai.lstm_ai import (
    load_metadata,
    predict_exercise
)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

metadata = load_metadata("./ai/models/bicep_curl/lstm_metadata.pkl")

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

    points = {}
    for i, (x, y, z) in enumerate(pose_points):
        points[f'x{i}'] = x
        points[f'y{i}'] = y
        points[f'z{i}'] = z

    print(f"✅ ID={exercise_id}, ได้ {len(pose_points)} จุด")
    print("จุดแรก:", pose_points[0])

    # model = load_stacking_model("./ai/models/bicep_curl_model.pkl")
    predicted_exercise = None
    bicep_curl = False
    lateral_raise = False
    squat = False
    if metadata:
        if exercise_id == "1":
            predicted_exercise = predict_exercise(points, metadata, model="bicep_curl")
            bicep_curl = predicted_exercise == "bicep_curl"
        elif exercise_id == "2":
            predicted_exercise = predict_exercise(points, metadata, model="lateral_raise")
            lateral_raise = predicted_exercise == "lateral_raise"
        elif exercise_id == "3":
            predicted_exercise = predict_exercise(points, metadata, model="squat")
            squat = predicted_exercise == "squat"
        print(f"{predicted_exercise=}")


    # คุณสามารถทำการบันทึก / ประมวลผลต่อได้ที่นี่
    return jsonify({"bicep": bicep_curl, "lateral": lateral_raise, "squat": squat})
    # return jsonify({"status": "received", "points": len(pose_points)})


    

if __name__ == "__main__":
    # socketio.run(app, host="127.0.0.1", port=5000, debug=True)
    app.run(
        host="0.0.0.0",
        port=8080,
        ssl_context=("secrets/ssl.crt", "secrets/ssl_private.key"),
        debug=True)
