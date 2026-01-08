from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Khởi tạo Flask app
app = Flask(__name__)

# Load model đã train sẵn
model = joblib.load("best_regression_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    rings = None
    age = None

    if request.method == "POST":
        data = {
            "Sex": request.form["Sex"],
            "Length": float(request.form["Length"]),
            "Diameter": float(request.form["Diameter"]),
            "Height": float(request.form["Height"]),
            "Whole weight": float(request.form["Whole_weight"]),
            "Shucked weight": float(request.form["Shucked_weight"]),
            "Viscera weight": float(request.form["Viscera_weight"]),
            "Shell weight": float(request.form["Shell_weight"]),
        }

        df = pd.DataFrame([data])
        rings = model.predict(df)[0]
        age = rings + 1.5

    return render_template(
        "index.html",
        rings=rings,
        age=age
    )

# Khi chạy trực tiếp bằng python app.py (local)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)