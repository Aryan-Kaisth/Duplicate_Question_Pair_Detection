from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from src.pipelines.prediction_pipeline import PredictionPipeline
from app.schema import QuestionPairInput, DuplicatePredictionOutput


app = FastAPI(title="Duplicate Question Detection")

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model 
prediction_pipeline = PredictionPipeline()
THRESHOLD = 0.52

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post(
    "/predict",
    response_model=DuplicatePredictionOutput
)
async def predict_duplicate(data: QuestionPairInput):

    # raw text inputs
    score = prediction_pipeline.predict(
        data.question1,
        data.question2,
    )

    is_duplicate = score >= THRESHOLD

    return DuplicatePredictionOutput(
        is_duplicate=is_duplicate,
        label="Duplicate question pair"
        if is_duplicate
        else "Not a duplicate question pair"
    )
