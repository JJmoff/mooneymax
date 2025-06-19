import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import asyncpg
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from io import BytesIO
from pydantic import BaseModel, ValidationError, Field
from sentence_transformers import SentenceTransformer # For generating embeddings
from unstructured.partition.auto import partition, PartitionError, FiletypeError # For document parsing

# Configure logging for the service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CreditReportIngestionService")

app = FastAPI(
    title="Credit Report Ingestion Service",
    description="Service to ingest, parse, embed, and store credit report documents.",
    version="1.0.0"
)

# --- Configuration Management ---
# It's highly recommended to use environment variables for production deployments
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")

# --- Global Resources ---
db_pool: Optional[asyncpg.Pool] = None
embedding_model: Optional[SentenceTransformer] = None
# ocr_model: Optional[Any] = None # Uncomment and configure if using PaddleOCR or other OCR tools

# --- Custom Exception Classes ---
class IngestionError(Exception):
    """Base exception for Ingestion Service errors."""
    pass

class InvalidFileFormatError(IngestionError):
    """Raised when the uploaded file format is not supported."""
    pass

class ParsingError(IngestionError):
    """Raised when an error occurs during parsing with Unstructured.io."""
    pass

class EmbeddingError(IngestionError):
    """Raised when an error occurs during embedding generation."""
    pass

class DatabaseInsertionError(IngestionError):
    """Raised when an error occurs during database insertion."""

# --- Pydantic Models for Data Validation and Structuring ---
class CreditReportSection(BaseModel):
    """Represents a structured section/element extracted from a credit report."""
    section_title: str = Field(..., description="Type of the element (e.g., 'Title', 'NarrativeText').")
    content: str = Field(..., description="Textual content of the section.")
    metadata: Dict[str, Any] = Field({}, description="Metadata associated with the section.")
    # embedding: List[float] = Field([], description="Vector embedding of the section's content (optional for sections).")
    # Removed section-level embedding for simplicity in this example, only full report embedding is stored

class CreditReportIngestionResponse(BaseModel):
    """Response model for a successful credit report upload and processing."""
    report_id: str = Field(..., description="Unique ID assigned to the ingested credit report.")
    user_id: int = Field(..., description="ID of the user who uploaded the report.")
    message: str = Field(..., description="Status message for the ingestion process.")
    filename: str = Field(..., description="Original filename of the uploaded report.")
    processing_timestamp: str = Field(..., description="Timestamp when the report was processed.")

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event() -> None:
    """Initializes database connection pool and loads AI models on application startup."""
    global db_pool, embedding_model #, ocr_model
    logger.info("Credit Report Ingestion Service: Starting up...")

    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Credit Report Ingestion Service: Database connection pool created successfully.")
    except Exception as e:
        logger.error(f"Credit Report Ingestion Service: Failed to create database connection pool: {e}", exc_info=True)
        raise # Critical error, prevent app from starting

    logger.info("Credit Report Ingestion Service: Loading embedding model 'all-MiniLM-L6-v2'...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Credit Report Ingestion Service: Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Credit Report Ingestion Service: Failed to load embedding model: {e}", exc_info=True)
        raise # Critical error, prevent app from starting

    # If you intend to use PaddleOCR, uncomment and configure this section
    # logger.info("Credit Report Ingestion Service: Loading OCR model...")
    # try:
    #     ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
    #     logger.info("Credit Report Ingestion Service: OCR model loaded successfully.")
    # except Exception as e:
    #     logger.error(f"Credit Report Ingestion Service: Failed to load OCR model: {e}", exc_info=True)
    #     raise # Critical error, prevent app from starting

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Closes database connection pool on application shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Credit Report Ingestion Service: Database connection pool closed.")
    logger.info("Credit Report Ingestion Service: Shut down complete.")

# --- API Endpoints ---
@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint for the Credit Report Ingestion Service."""
    return {"message": "Welcome to the Credit Report Ingestion Service!"}

@app.post("/upload_report/", response_model=CreditReportIngestionResponse)
async def upload_report_endpoint(
    file: UploadFile = File(..., description="The credit report file to upload (PDF, DOCX, JPG, PNG, etc.)."),
    user_id: int = Form(..., description="The ID of the user associated with this credit report.")
) -> CreditReportIngestionResponse:
    """
    Uploads a credit report, processes its content, generates embeddings,
    and stores the structured data and embeddings in the database.
    """
    report_filename = file.filename
    logger.info(f"Received upload request for file: '{report_filename}' for user ID: {user_id}")

    try:
        file_content = await file.read()
        
        # Ensure embedding model is available
        if embedding_model is None:
            logger.error("Embedding model is not initialized.")
            raise EmbeddingError("Embedding model not available. Service may not have started correctly.")

        # --- File Parsing with Unstructured.io ---
        elements = []
        try:
            # Use BytesIO to pass file content to unstructured.partition
            elements = partition(
                file=BytesIO(file_content),
                filename=report_filename,
                strategy="hi_res", # 'hi_res' uses OCR for images/scanned PDFs
                ocr_languages="eng", # Specify OCR language
                # If using PaddleOCR via 'unstructured', you might need to pass ocr_model
                # ocr_agent=ocr_model if ocr_model else None
            )
            logger.info(f"Successfully partitioned '{report_filename}'. Extracted {len(elements)} elements.")
        except FiletypeError as e:
            logger.warning(f"Invalid file type uploaded for '{report_filename}': {e}")
            raise InvalidFileFormatError(f"Unsupported file type. Please upload a common document or image format.")
        except PartitionError as e:
            logger.error(f"Failed to partition '{report_filename}': {e}", exc_info=True)
            raise ParsingError(f"Error processing document structure: {e}. The file might be corrupted or malformed.")
        except Exception as e: # Catch any other unexpected errors from partition
            logger.error(f"Unexpected error during partitioning of '{report_filename}': {e}", exc_info=True)
            raise ParsingError(f"An unexpected error occurred during document parsing: {e}")

        # --- Extract and Validate Text ---
        full_text = "\n".join([str(element.text) for element in elements if element.text is not None])

        if not full_text.strip():
            logger.warning(f"No meaningful text extracted from '{report_filename}'.")
            # Return a 200 OK with a specific message for no content extracted
            return CreditReportIngestionResponse(
                report_id=str(uuid.uuid4()), # Still generate an ID even if empty content
                user_id=user_id,
                message="File processed, but no readable text content was extracted. It might be empty or unreadable.",
                filename=report_filename,
                processing_timestamp=datetime.now(timezone.utc).isoformat()
            )

        # --- Pydantic Validation and Structuring for Sections ---
        processed_sections: List[CreditReportSection] = []
        try:
            for element in elements:
                # Ensure element.text is not None before processing
                if element.text:
                    section_metadata = element.metadata.to_dict() if hasattr(element.metadata, 'to_dict') else dict(element.metadata)
                    processed_sections.append(CreditReportSection(
                        section_title=str(element.type), # Convert type to string
                        content=element.text,
                        metadata=section_metadata,
                        # Section embeddings could be added here if needed for finer-grained RAG
                        # embedding=embedding_model.encode(element.text).tolist() if embedding_model else []
                    ))
            logger.info(f"Structured {len(processed_sections)} sections for '{report_filename}'.")
        except ValidationError as e:
            logger.error(f"Data validation failed for '{report_filename}' during section processing: {e}", exc_info=True)
            raise ParsingError(f"Internal data structuring error after parsing: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during section processing for '{report_filename}': {e}", exc_info=True)
            raise ParsingError(f"An unexpected error occurred during internal data structuring: {e}")

        # --- Embedding Generation for the Full Report ---
        report_embedding: List[float]
        try:
            report_embedding = embedding_model.encode(full_text).tolist()
            logger.info(f"Generated embedding for full report '{report_filename}'.")
        except Exception as e:
            logger.error(f"Error generating embedding for full report '{report_filename}': {e}", exc_info=True)
            raise EmbeddingError(f"Failed to generate embedding for the report: {e}")

        # --- Database Insertion ---
        if db_pool is None:
            logger.error("Database connection pool is not initialized.")
            raise DatabaseInsertionError("Database service is not available.")

        generated_report_id = str(uuid.uuid4())
        current_timestamp = datetime.now(timezone.utc).isoformat()

        try:
            async with db_pool.acquire() as connection:
                # Store the entire processed structure as JSONB, and the full report embedding
                await connection.execute(
                    """
                    INSERT INTO credit_reports (report_id, user_id, filename, processing_timestamp, structured_data, embedding)
                    VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector);
                    """,
                    generated_report_id,
                    user_id,
                    report_filename,
                    current_timestamp,
                    json.dumps([section.model_dump() for section in processed_sections]), # Convert list of Pydantic models to JSON string
                    report_embedding
                )
            logger.info(f"Successfully inserted data for '{report_filename}' into database with Report ID: {generated_report_id}.")
            return CreditReportIngestionResponse(
                report_id=generated_report_id,
                user_id=user_id,
                message="Credit report uploaded and processed successfully.",
                filename=report_filename,
                processing_timestamp=current_timestamp
            )
        except Exception as e:
            logger.error(f"Error inserting data for '{report_filename}' into the database: {e}", exc_info=True)
            raise DatabaseInsertionError(f"Failed to save processed report to database: {e}")

    # --- Top-Level Exception Handling for HTTP Responses ---
    except InvalidFileFormatError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except (ParsingError, EmbeddingError, DatabaseInsertionError) as e:
        # These are internal processing failures, map to 500
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Report processing failed: {e}")
    except HTTPException as e:
        # Re-raise HTTPExceptions that were intentionally raised
        raise e
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"An unhandled error occurred during file processing for '{report_filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected internal server error occurred.")

if __name__ == "__main__":
    import uvicorn
    # Set dummy environment variables for local testing
    os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")
    uvicorn.run(app, host="0.0.0.0", port=8001) # Credit Report Ingestion Service runs on port 8001
