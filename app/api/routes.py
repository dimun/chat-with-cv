from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from app.services.vector_db import document_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the document
    document_retriever.load_document("cv.md")
    yield
    pass


app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "v0.0.1"}

@app.post("/embed_and_store")
async def embed_and_store():
    return {"message": "ok"}

@app.get("/handle_query")
async def handle_query(q: str = Query(default="")):
    return {"message": document_retriever.retrieve_document(q)}

