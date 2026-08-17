import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="API Predicción de Viviendas")

# Obtener la ruta absoluta del directorio actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "modelo_viviendas.joblib")
OPCIONES_PATH = os.path.join(BASE_DIR, "opciones.json")

# Cargar modelo y opciones usando rutas absolutas
try:
  modelo = joblib.load(MODEL_PATH)
  print("Modelo cargado exitosamente.")
except Exception as e:
  modelo = None
  print(f"Error al cargar el modelo: {e}")

try:
  with open(OPCIONES_PATH, "r", encoding="utf-8") as f:
    opciones_dataset = json.load(f)
except Exception as e:
  opciones_dataset = {"municipios": [], "inmobiliarias": []}
  print(f"Error al cargar opciones: {e}")


class ViviendaInput(BaseModel):
  habitaciones: int
  banos: float
  area_construida: float
  area_total: float
  parqueaderos: float
  municipio: str
  inmobiliaria: str = "Desconocido"


@app.get("/")
def home():
  return {"mensaje": "API de Predicción de Precios Activa"}


@app.get("/opciones")
def obtener_opciones():
  return opciones_dataset


@app.post("/predict")
def predecir_precio(datos: ViviendaInput):
  if modelo is None:
    raise HTTPException(
        status_code=500,
        detail="Modelo no cargado. Verifica los logs del servidor.",
    )

  df_entrada = pd.DataFrame([{
      "Habitaciones": datos.habitaciones,
      "Banos": datos.banos,
      "Area_Construida": datos.area_construida,
      "Area_Total": datos.area_total,
      "Parqueaderos": datos.parqueaderos,
      "Municipio": datos.municipio,
      "Inmobiliaria": datos.inmobiliaria,
  }])

  precio_estimado = float(modelo.predict(df_entrada)[0])

  return {
      "status": "exito",
      "precio_estimado_cop": round(precio_estimado, 2),
      "moneda": "COP",
  }