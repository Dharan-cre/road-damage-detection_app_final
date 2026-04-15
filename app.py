from flask import Flask, render_template, request, redirect, url_for, session
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CNN Model
cnn_model = load_model("final_model.keras")

# LOAD 3 YOLO MODELS
crack_model = YOLO("models/crack_best.pt")
flood_model = YOLO("models/flood_best.pt")
pothole_model = YOLO("models/pothole_best.pt")

class_names = [
    "Cracks",
    "No Cracks",
    "Water Logging",
    "No Potholes",
    "Potholes",
    "Water_logging"
]

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["user"] = "admin"
            return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    cnn_result = yolo_result = confidence = recommendation = None
    image_path = yolo_image_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_path = filepath

            # ================= CNN =================
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = cnn_model.predict(img_array)
            idx = np.argmax(pred)
            confidence = round(np.max(pred) * 100, 2)
            cnn_result = class_names[idx]

            # ================= YOLO MULTI-MODEL =================
            img_cv = cv2.imread(filepath)
            all_detections = []

            def run_model(model, label_name, color, conf_thres=0.3):
                results = model(filepath)
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            if conf > conf_thres:
                                all_detections.append((x1, y1, x2, y2, label_name, conf, color))
                    elif r.masks is not None:
                        for seg in r.masks.xy:
                            x_coords = seg[:, 0]
                            y_coords = seg[:, 1]
                            x1, y1 = int(min(x_coords)), int(min(y_coords))
                            x2, y2 = int(max(x_coords)), int(max(y_coords))
                            all_detections.append((x1, y1, x2, y2, label_name, 0.5, color))

            run_model(crack_model, "Crack", (0, 255, 0), 0.2)
            run_model(pothole_model, "Pothole", (0, 0, 255), 0.3)
            run_model(flood_model, "Water", (255, 0, 0), 0.1)

            # ================= DRAW BOXES =================
            detected_labels = []
            for (x1, y1, x2, y2, label, conf, color) in all_detections:
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(img_cv, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                detected_labels.append(label)

            if detected_labels:
                yolo_result = ", ".join(set(detected_labels))
            else:
                yolo_result = "No damage detected"

            # ================= SAVE OUTPUT =================
            yolo_output_path = os.path.join(UPLOAD_FOLDER, "yolo_output.jpg")
            cv2.imwrite(yolo_output_path, img_cv)
            yolo_image_path = yolo_output_path

            # ================= RECOMMENDATION =================
            if "Crack" in yolo_result or "Pothole" in yolo_result:
                recommendation = "⚠️ Road damage detected"
            elif "Water" in yolo_result:
                recommendation = "⚠️ Water logging detected"
            else:
                recommendation = "✅ Road condition good"

    return render_template(
        "dashboard.html",
        cnn_result=cnn_result,
        yolo_result=yolo_result,
        confidence=confidence,
        image_path=image_path,
        yolo_image_path=yolo_image_path,
        recommendation=recommendation
    )


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=7860)
