from fastapi import FastAPI
from pydantic import BaseModel

from src.inference import VLLMInference

app = FastAPI(title="vLLM Inference API")
model = VLLMInference()


class QueryRequest(BaseModel):
    query: str


@app.post("/generate")
def generate(request: QueryRequest):
    result = model.generate(request.query)
    return {
        "query": request.query,
        "response": result["response"],
        "elapsed_ms": result["elapsed_ms"],
    }
