from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.chunker import split_text_into_chunks
from app.services.document_processor import extract_text_from_pdf
from app.services.embeddings import create_embeddings
from app.services.vector_store import save_vector_store


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    file_content = await file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty",
        )

    try:
        extracted_text, page_count = extract_text_from_pdf(file_content)

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the PDF",
            )

        chunks = split_text_into_chunks(extracted_text)
        embeddings = create_embeddings(chunks)

        stored_vectors = save_vector_store(
            embeddings=embeddings,
            chunks=chunks,
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"The PDF could not be processed: {error}",
        ) from error

    return {
        "filename": file.filename,
        "pages": page_count,
        "characters": len(extracted_text),
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
        "stored_vectors": stored_vectors,
        "message": "Document processed and stored successfully",
    }