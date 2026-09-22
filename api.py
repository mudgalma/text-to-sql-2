import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict
from dotenv import load_dotenv

from main import TextToSQLEngine

load_dotenv()

# Global state for the engine
engine: TextToSQLEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Initialize engine on startup
    engine = TextToSQLEngine()
    yield
    # Close resources on shutdown
    if engine:
        engine.close()

app = FastAPI(title="Text-to-SQL API", lifespan=lifespan)

# Allow CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    generated_logic: str | None
    result: list[Dict[str, Any]] | None
    confidence_score: float
    explanation: str

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a natural language query and return the result."""
    result = engine.run_query(request.query)
    return result
