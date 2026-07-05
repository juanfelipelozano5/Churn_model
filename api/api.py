from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(
    title="Churn Prediction API",
    description="Microservicio para predecir probabilidad de fuga de clientes bancarios",
    version="1.0.0"
)

modelo = joblib.load("modelo_churn.pkl")
scaler = joblib.load("scaler_churn.pkl")

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForest", "version": "1.0.0"}

@app.post("/predict")
def predict(data: dict):
    """
    Recibe un JSON con las features del cliente y retorna
    la probabilidad de churn, la clasificación y el umbral usado.

    Ejemplo de request:
    {
        "features": [619, 42, 2, 0.0, 1, 1, 1, 101348.88, 0.0, 0.047, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    }
    """
    X = np.array(data["features"]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    proba = modelo.predict_proba(X_scaled)[0][1]
    umbral = 0.3

    return {
        "probabilidad_churn": round(float(proba), 4),
        "clasificacion": "ALTO RIESGO" if proba >= umbral else "BAJO RIESGO",
        "umbral_usado": umbral,
        "interpretacion": "Cliente con alta probabilidad de fuga. Priorizar contacto de retención."
                          if proba >= umbral else
                          "Cliente con baja probabilidad de fuga. Monitoreo estándar."
    }
