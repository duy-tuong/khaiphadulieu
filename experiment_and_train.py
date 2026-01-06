import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. Đọc dữ liệu
# =========================
data = pd.read_csv("abalone.csv")

X = data.drop("Rings", axis=1)
y = data["Rings"]

# =========================
# 2. Tiền xử lý dữ liệu
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
        ("num", "passthrough", X.columns.drop("Sex"))
    ]
)

# =========================
# 3. Chia dữ liệu
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Các mô hình hồi quy
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = {}

# =========================
# 5. Thực nghiệm
# =========================
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = mae

    print("=================================")
    print(f"Model: {name}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

# =========================
# 6. Chọn mô hình tốt nhất
# =========================
best_model_name = min(results, key=results.get)
print("\n>>> Mô hình tốt nhất:", best_model_name)

best_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", models[best_model_name])
])

best_pipeline.fit(X_train, y_train)
joblib.dump(best_pipeline, "best_regression_model.pkl")

print("Đã lưu best_regression_model.pkl")
