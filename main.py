import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Inicializar la aplicación de FastAPI
app = FastAPI(title="API Predicción de Viviendas")

# Configurar middleware de CORS (Solución al error "Failed to fetch")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Obtener la ruta absoluta del directorio actual para evitar errores en Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "modelo_viviendas.joblib")
OPCIONES_PATH = os.path.join(BASE_DIR, "opciones.json")

# Cargar el modelo de machine learning
try:
  modelo = joblib.load(MODEL_PATH)
  print("Modelo cargado exitosamente.")
except Exception as e:
  modelo = None
  print(f"Error al cargar el modelo: {e}")

# Cargar las opciones del JSON
try:
  with open(OPCIONES_PATH, "r", encoding="utf-8") as f:
    opciones_dataset = json.load(f)
except Exception as e:
  opciones_dataset = {"municipios": [], "inmobiliarias": []}
  print(f"Error al cargar opciones: {e}")


# Esquema de datos de entrada con Pydantic
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
        detail=(
            "Modelo no cargado. Revisa los logs del servidor para más detalles."
        ),
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