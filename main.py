from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ✅ Load the trained model
MODEL_PATH = "phish_detector.h5"  # Updated model path
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load the saved TF-IDF Vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Define request body format
class URLInput(BaseModel):
    url: str

# ✅ Prediction route
@app.post("/predict")
def predict(data: URLInput):
    url = [data.url]  # Convert input into a list for TF-IDF transformation
    features = vectorizer.transform(url)  # Convert URL to numerical data
    features = features.toarray()  # Convert sparse matrix to dense

    # Make a prediction
    prediction = model.predict(features)[0][0]  # Get probability
    result = "phishing" if prediction > 0.5 else "legit"

    return {
        "url": data.url,
        "phishing_probability": float(prediction),
        "prediction": result
    }

# ✅ Root route
@app.get("/")
def home():
    return {"message": "Phishing URL Detection API is running!"}
