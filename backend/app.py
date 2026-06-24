from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'mnist_model.keras')

model = load_model(MODEL_PATH)


@app.get('/')
def home():
    return {"message": "MNIST API Running"}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image = image.convert("L")

    image = image.resize((28, 28))

    image_array = np.array(image)

    image_array = image_array.astype("float32") / 255

    image_array = image_array.reshape(1, 784)

    prediction = np.argmax(
        model.predict(image_array),
        axis=1
    )[0]

    return {
        "prediction": int(prediction)
    }