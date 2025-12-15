from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from src.pipelines.prediction_pipeline import PredictionPipeline
from app.schema import QuestionPairInput, DuplicatePredictionOutput

app = FastAPI(title="Duplicate Question Detection")

# Static files (CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load model once at startup
prediction_pipeline = PredictionPipeline()
THRESHOLD = 0.54  # your tuned threshold


# ------------------------
# Home Page
# ------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ------------------------
# API Prediction Endpoint
# ------------------------
@app.post("/predict")
async def predict_duplicate(data: QuestionPairInput):
    score = prediction_pipeline.predict(
        data.question1,
        data.question2
    )

    is_duplicate = score >= THRESHOLD

    return {
        "is_duplicate": is_duplicate
    }
