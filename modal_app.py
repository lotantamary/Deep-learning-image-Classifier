import io
import json

import modal

# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------

app = modal.App("dog-breed-classifier")

# Container image with all Python dependencies + model files baked in
image = (
    modal.Image.debian_slim()
    .pip_install(
        "tensorflow==2.20.0",
        "tensorflow-hub==0.16.1",
        "tf-keras==2.20.1",
        "numpy",
        "pillow",
        "fastapi",
        "python-multipart",
    )
    .add_local_dir("models/", remote_path="/models")
)

# ---------------------------------------------------------------------------
# Classifier class — model loaded once per container (not on every request)
# ---------------------------------------------------------------------------

@app.cls(image=image)
class Classifier:

    @modal.enter()
    def load_model(self):
        import numpy as np
        import tensorflow_hub as hub
        import tf_keras as keras
        from PIL import Image

        self.np = np
        self.Image = Image
        self.model = keras.models.load_model(
            "/models/mobilenet_v2_dog_breed_classifier.h5",
            custom_objects={"KerasLayer": hub.KerasLayer},
        )
        with open("/models/labels.json") as f:
            self.labels = json.load(f)

    @modal.method()
    def predict(self, image_bytes: bytes) -> list[dict]:
        img = self.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))

        img_array = self.np.array(img, dtype=self.np.float32) / 255.0
        img_array = self.np.expand_dims(img_array, axis=0)

        probs = self.model.predict(img_array, verbose=0)[0]
        top3_indices = self.np.argsort(probs)[-3:][::-1]

        return [
            {
                "breed": self.labels[i].replace("_", " ").title(),
                "confidence": round(float(probs[i]) * 100, 1),
            }
            for i in top3_indices
        ]


# ---------------------------------------------------------------------------
# HTTP endpoint — all fastapi imports are inside this function so they are
# only resolved inside the Modal container (not needed locally).
# ---------------------------------------------------------------------------

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    web_app = FastAPI(title="Dog Breed Classifier API")

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @web_app.post("/predict")
    async def predict_endpoint(file: UploadFile = File(...)):
        image_bytes = await file.read()
        classifier = Classifier()
        predictions = classifier.predict.remote(image_bytes)
        return JSONResponse({"predictions": predictions})

    return web_app
