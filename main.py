import json
import os
import joblib
import traceback
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Inicializar la aplicación FastAPI
app = FastAPI(
    title="API Predicción de Viviendas",
    description="API robusta para la estimación de precios de vivienda con manejo integrado de errores.",
    version="1.0.0"
)

# 2. Permitir CORS (Evita bloqueos de 'Failed to fetch' en Swagger UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Definición de rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_viviendas.joblib")
OPCIONES_PATH = os.path.join(BASE_DIR, "opciones.json")

# Variables globales para almacenamiento en memoria
modelo = None
opciones_dataset = {"municipios": [], "inmobiliarias": []}
estado_carga = {"modelo_cargado": False, "error_modelo": None}

# 4. Carga segura del modelo de Machine Learning
try:
    if os.path.exists(MODEL_PATH):
        modelo = joblib.load(MODEL_PATH)
        estado_carga["modelo_cargado"] = True
        print("✅ Modelo cargado correctamente.")
    else:
        estado_carga["error_modelo"] = f"El archivo no existe en la ruta: {MODEL_PATH}"
        print(f"❌ {estado_carga['error_modelo']}")
except Exception as e:
    estado_carga["error_modelo"] = f"Excepción al cargar el modelo: {str(e)}"
    print(f"❌ {estado_carga['error_modelo']}")

# 5. Carga segura del archivo JSON de opciones
try:
    if os.path.exists(OPCIONES_PATH):
        with open(OPCIONES_PATH, "r", encoding="utf-8") as f:
            opciones_dataset = json.load(f)
        print("✅ Opciones cargadas correctamente.")
    else:
        print(f"⚠️ No se encontró opciones.json en {OPCIONES_PATH}")
except Exception as e:
    print(f"⚠️ Error leyendo opciones.json: {str(e)}")


# 6. Esquema de Validación de Datos con Pydantic
class ViviendaInput(BaseModel):
    habitaciones: int = Field(..., example=3, description="Número de habitaciones")
    banos: float = Field(..., example=2.0, description="Número de baños")
    area_construida: float = Field(..., example=85.0, description="Área construida en m²")
    area_total: float = Field(..., example=90.0, description="Área total del predio en m²")
    parqueaderos: float = Field(..., example=1.0, description="Número de parqueaderos")
    municipio: str = Field(..., example="Sabaneta", description="Municipio de ubicación")
    inmobiliaria: str = Field(default="Desconocido", example="Desconocido", description="Inmobiliaria encargada")


# 7. Endpoints de Diagnóstico e Información

@app.get("/", summary="Endpoint Raíz")
def home():
    """Retorna el estado general de la API."""
    return {
        "status": "activa",
        "modelo_cargado": estado_carga["modelo_cargado"],
        "mensaje": "API de Predicción de Precios de Vivienda"
    }

@app.get("/health", summary="Diagnóstico del Sistema")
def health_check():
    """Permite verificar el estado exacto del servidor y rutas de archivos."""
    return {
        "directorio_base": BASE_DIR,
        "ruta_modelo": MODEL_PATH,
        "modelo_existe_en_disco": os.path.exists(MODEL_PATH),
        "modelo_cargado_en_memoria": estado_carga["modelo_cargado"],
        "error_carga": estado_carga["error_modelo"]
    }

@app.get("/opciones", summary="Obtener lista de Opciones")
def obtener_opciones():
    """Devuelve los valores válidos para municipio e inmobiliaria."""
    return opciones_dataset


# 8. Endpoint de Predicción Defensivo

@app.post("/predict", summary="Predecir Precio de Vivienda")
def predecir_precio(datos: ViviendaInput):
    """
    Recibe las características de una vivienda y devuelve el precio estimado en COP.
    Maneja excepciones internas para detallar si faltan columnas o fallan los tipos de datos.
    """
    # Validación de la carga del modelo
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "El modelo no está disponible.",
                "causa": estado_carga["error_modelo"],
                "sugerencia": "Verifica que el archivo 'modelo_viviendas.joblib' esté en la raíz del repositorio."
            }
        )

    try:
        # Mapeo de campos a la estructura requerida por el modelo
        datos_dict = {
            "Habitaciones": datos.habitaciones,
            "Banos": datos.banos,
            "Area_Construida": datos.area_construida,
            "Area_Total": datos.area_total,
            "Parqueaderos": datos.parqueaderos,
            "Municipio": datos.municipio,
            "Inmobiliaria": datos.inmobiliaria,
        }

        df_entrada = pd.DataFrame([datos_dict])

        # Realizar predicción
        prediccion = modelo.predict(df_entrada)
        precio_estimado = float(prediccion[0])

        return {
            "status": "exito",
            "precio_estimado_cop": round(precio_estimado, 2),
            "moneda": "COP",
            "datos_recibidos": datos_dict
        }

    except KeyError as e:
        # Error cuando falta una columna requerida por el modelo
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Columna faltante en la entrada.",
                "columna_afectada": str(e),
                "mensaje": "Compara las claves del dictionary en main.py con las columnas usadas durante el entrenamiento."
            }
        )
    except Exception as e:
        # Captura de cualquier error en el Pipeline de ML (ej. categorías no vistas)
        print("Traceback del error:\n", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Fallo al ejecutar la predicción.",
                "detalle_tecnico": str(e),
                "tipo_error": type(e).__name__
            }
        )