from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
OUTPUT_CSV = "data/result.csv"


@app.get("/", response_class=HTMLResponse)
def read_data(request: Request):
    if not os.path.exists(OUTPUT_CSV):
        return templates.TemplateResponse("index.html", {"request": request, "table": "No data yet."})

    df = pd.read_csv(OUTPUT_CSV)
    return templates.TemplateResponse("index.html", {"request": request, "table": df.to_html(index=False)})
