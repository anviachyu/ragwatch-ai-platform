from fastapi import FastAPI

from app.api.documents import router as documents_router


app = FastAPI(
    title="RAGWatch API",
    description="Document question-answering API with source citations",
    version="1.0.0",
)

app.include_router(documents_router)


@app.get("/")
def root():
    return {
        "name": "RAGWatch",
        "message": "RAGWatch API is running",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}