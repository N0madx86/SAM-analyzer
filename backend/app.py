from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from models.inference_engine import SentimentEngine


app = FastAPI(
    title="SAM Sentiment Analysis API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load models once when the API starts
engine = SentimentEngine()


class PredictionRequest(BaseModel):

    text: str


@app.get("/")
def root():

    return {
        "status": "online",
        "service": "SAM Sentiment Analysis API"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    result = engine.predict(
        request.text
    )

    return result