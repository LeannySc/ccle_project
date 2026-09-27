import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import joblib

# 1. Cargar variables de entorno desde el archivo .env
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# 2. Construir conexión SQLAlchemy usando psycopg2
connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

print("Conectando a la base de datos de la torre CCLE vía Tailscale...")

# 3. Extraer los datos a un DataFrame de Pandas
query = "SELECT * FROM base_mercado;"
df = pd.read_sql(query, engine)

print(f"¡Datos extraídos con éxito en tu PC! Registros obtenidos: {len(df)}")

# 4. Preparación de Variables (Features y Target)
target_column = "precio_pesos"

# Limpiar registros donde el precio objetivo sea nulo
df = df.dropna(subset=[target_column])

# Seleccionar variables categóricas clave para predecir el precio
cat_features = ["tipo_producto", "presentacion", "unidad_medida", "departamento"]
cat_features = [col for col in cat_features if col in df.columns]

X = df[cat_features]
y = df[target_column]

# 5. Crear el Pipeline de Preprocesamiento y Modelo
# OneHotEncoder transforma las categorías de texto en columnas numéricas binarias
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='drop'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 6. División de datos (Train/Test) y Entrenamiento Local
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Entrenando el modelo de Machine Learning utilizando los recursos de tu PC...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluación rápida
y_pred = model_pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"¡Modelo entrenado con éxito! Coeficiente de determinación (R²): {score:.4f}")

# 8. Guardar el modelo entrenado en un archivo binario (.joblib)
os.makedirs("ml_models/models", exist_ok=True)
model_filename = "ml_models/models/modelo_mercado.joblib"
joblib.dump(model_pipeline, model_filename)

print(f"Modelo empaquetado y guardado localmente en: {model_filename}")