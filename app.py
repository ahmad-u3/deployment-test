import json
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ARTIFACT_DIR = Path('artifacts/house_price_v1')
model = joblib.load(ARTIFACT_DIR / 'model.joblib')
metadata = json.loads((ARTIFACT_DIR / 'metadata.json').read_text())

app = FastAPI(title='House Price API', version=metadata['version'])


class House(BaseModel):
    crime_rate: float = Field(ge=0)
    resid_area: float
    air_qual: float
    room_num: float = Field(gt=0)
    age: float = Field(ge=0, le=100)
    dist1: float
    dist2: float
    dist3: float
    dist4: float
    teachers: float
    poverty_rate: float
    airport: Literal['YES', 'NO']
    hospital_beds: Optional[float] = None
    hotel_rooms: float
    waterbody: Optional[Literal['Lake', 'River', 'Lake and River']] = None
    rainfall: float
    has_bus_terminal: Literal['YES', 'NO'] = 'YES'
    parks: float


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.get('/model-info')
def model_info():
    return metadata


@app.post('/predict')
def predict(house: House):
    X = pd.DataFrame([house.model_dump()])
    price = float(model.predict(X)[0])
    return {'predicted_price': round(price, 2), 'model_version': metadata['version']}


@app.post('/predict/batch')
def predict_batch(houses: List[House]):
    X = pd.DataFrame([h.model_dump() for h in houses])
    prices = model.predict(X)
    return {'predicted_prices': [round(float(p), 2) for p in prices],
            'model_version': metadata['version']}
