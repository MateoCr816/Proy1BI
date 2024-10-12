from fastapi import FastAPI, Request
from joblib import dump, load
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

app = FastAPI()

#Ruta del pipeline
model_path = 'assets/modelo_nuevo.joblib'

@app.post("/reentrenar")
async def reentrenar_modelo(request: Request):
    # Datos para entrenar el modelo
    request_data = await request.json()

    # Extraer los textos y las etiquetas sdg de los datos
    X_nuevos = pd.DataFrame([item['Textos_espanol'] for item in request_data], columns=["Textos_espanol"])
    y_nuevos = pd.Series([item['sdg'] for item in request_data])

    # Division los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_nuevos, y_nuevos, test_size=0.2, random_state=42)

    # Uso del pipeline
    pipeline = load(model_path)

    # Entrenar el modelo con los datos nuevos
    pipeline.fit(X_train["Textos_espanol"], y_train)

    # Hacer predicciones en los datos de prueba
    predictions = pipeline.predict(X_test["Textos_espanol"])

    # Calcular de las metricas
    accuracy = accuracy_score(y_test, predictions)
    matrix_confusion = confusion_matrix(y_test, predictions)
    classification_rpt = classification_report(y_test, predictions, output_dict=True)
    filtered_classification_rpt = {key: classification_rpt[key] for key in ['3', '4', '5']}

    # Guardar el modelo entrenado
    dump(pipeline, model_path)

    # Retornar las métricas de evaluación JSON
    return {
        "Accuracy": accuracy,
        "Matriz de Confusión": matrix_confusion.tolist(),
        "Reporte de Clasificación": filtered_classification_rpt
    }

@app.post("/predict")
async def make_predictions(request: Request):
    # Cargar el pipeline
    model = load(model_path)

    # Recibir los datos de prueba del modelo de clasificación
    request_data = await request.json()
    X_input = pd.DataFrame([item['Textos_espanol'] for item in request_data], columns=["Textos_espanol"])

    # HPredicciones
    result = model.predict(X_input["Textos_espanol"])

    # Retorno de la clasificación
    return {"sdg": result.tolist()}
