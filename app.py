import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age",
]
Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
log_lock = threading.Lock()


class ConcreteSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cement: Nonnegative = Field(description="Цемент, кг/м³")
    slag: Nonnegative = Field(description="Доменный шлак, кг/м³")
    fly_ash: Nonnegative = Field(description="Зола-унос, кг/м³")
    water: Nonnegative = Field(description="Вода, кг/м³")
    superplasticizer: Nonnegative = Field(description="Суперпластификатор, кг/м³")
    coarse_aggregate: Nonnegative = Field(description="Крупный заполнитель, кг/м³")
    fine_aggregate: Nonnegative = Field(description="Мелкий заполнитель, кг/м³")
    age: Positive = Field(description="Возраст, дни")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[ConcreteSample] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    prediction_time_ms: float
    target_unit: str = "MPa"
    request_id: str


def read_model(model_dir):
    metadata = json.loads((model_dir / "metadata.json").read_text())
    actual_hash = hashlib.sha256((model_dir / "model.pkl").read_bytes()).hexdigest()
    if actual_hash != metadata["model_sha256"]:
        raise ValueError("Model SHA-256 does not match metadata")
    if metadata["feature_names"] != FEATURES:
        raise ValueError("Model feature order does not match API contract")
    if metadata.get("task") != "regression" or metadata.get("target_unit") != "MPa":
        raise ValueError("This API requires a concrete-strength regression model in MPa")
    return mlflow.sklearn.load_model(str(model_dir)), metadata


@asynccontextmanager
async def lifespan(app):
    model_dir = Path(os.environ.get("MODEL_DIR", "runtime/model"))
    app.state.model, app.state.metadata = read_model(model_dir)
    app.state.log_path = Path(os.environ.get("PREDICTION_LOG", "logs/predictions.jsonl"))
    app.state.log_path.parent.mkdir(parents=True, exist_ok=True)
    yield
    app.state.model = None


app = FastAPI(
    title="Concrete Strength API",
    description="Лабораторная работа 4, вариант 14. Прогноз прочности бетона в МПа.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": str(app.state.metadata["model_version"]),
        "model_sha256": app.state.metadata["model_sha256"],
    }


@app.get("/metadata")
def metadata():
    return {
        **app.state.metadata,
        "max_batch_size": 1000,
        "input_constraints": "Числа >= 0; age > 0; строки, bool, NaN и Infinity запрещены",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictionRequest):
    instances = [row.model_dump() for row in body.instances]
    frame = pd.DataFrame(instances, columns=FEATURES)
    started = time.perf_counter()
    predictions = np.asarray(app.state.model.predict(frame), dtype=float)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if predictions.shape != (len(instances),) or not np.isfinite(predictions).all():
        raise RuntimeError("Model produced invalid predictions")
    response = PredictionResponse(
        predictions=predictions.tolist(),
        model_version=str(app.state.metadata["model_version"]),
        prediction_time_ms=elapsed_ms,
        request_id=str(uuid.uuid4()),
    )
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "instances": instances,
        **response.model_dump(),
        "model_sha256": app.state.metadata["model_sha256"],
    }
    with log_lock:
        with app.state.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return response


if __name__ == "__main__":
    import argparse
    import tempfile
    from fastapi.testclient import TestClient

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    payload = json.loads(Path("runtime/request.json").read_text())
    expected = json.loads(Path("runtime/expected.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        os.environ["PREDICTION_LOG"] = str(Path(directory) / "predictions.jsonl")
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            np.testing.assert_allclose(response.json()["predictions"], expected["predictions"], rtol=1e-10, atol=1e-10)
            assert response.json()["model_version"] == str(expected["model_version"])
            assert client.get("/health").json()["model_sha256"] == expected["model_sha256"]
            assert client.post("/predict", json={"instances": []}).status_code == 422
    print("OK: predictions, model version, SHA-256 and invalid request")
