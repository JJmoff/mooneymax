import os
import httpx
import asyncpg
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, EmailStr # Added EmailStr for type hinting

# LangChain and LLM imports for Gemini interaction (kept for /api/generate)
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure logging for the service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MooneyAIAgentService")

app = FastAPI(
    title="Mooney AI Agent Service",
    description="Backend service for handling Dialogflow CX webhooks, asynchronous updates, progress tracking, and LLM text generation.",
    version="1.0.0"
)

# --- Configuration Management ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://localhost:8004") # For now, points to itself for notifications
API_GATEWAY_NOTIFICATION_ENDPOINT = os.getenv("API_GATEWAY_NOTIFICATION_ENDPOINT", "/update_status") # Explicitly point to /update_status
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "TODO_YOUR_GEMINI_API_KEY") # Google Gemini API Key

# New: URL for the Credit Dispute Service
CREDIT_DISPUTE_SERVICE_URL = os.getenv("CREDIT_DISPUTE_SERVICE_URL", "http://localhost:8002")

# New: URL for the Email Service
EMAIL_SERVICE_URL = os.getenv("EMAIL_SERVICE_URL", "http://localhost:8000") # Points to the new Email Service

# --- Database Connection Pool (Global) ---
db_pool: Optional[asyncpg.Pool] = None

@app.on_event("startup")
async def startup_event() -> None:
    """Establishes the database connection pool on application startup."""
    global db_pool
    logger.info("Mooney AI Agent Service: Starting up...")
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        logger.info("Mooney AI Agent Service: Database connection pool created successfully.")
    except Exception as e:
        logger.error(f"Mooney AI Agent Service: Failed to create database connection pool: {e}", exc_info=True)
        raise # Re-raise to prevent app from starting without critical dependency

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Closes the database connection pool on application shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Mooney AI Agent Service: Database connection pool closed.")
    logger.info("Mooney AI Agent Service: Shut down complete.")

# --- Pydantic Models ---
class DialogflowRequest(BaseModel):
    fulfillmentInfo: Dict[str, Any] = Field({}, description="Fulfillment information, including intent tag.")
    sessionInfo: Dict[str, Any] = Field({}, description="Session information, including parameters.")
    text: Optional[str] = Field(None, description="Raw text query from the user.")
    queryInput: Dict[str, Any] = Field({}, description="Detailed query input.")
    detectIntentResponseId: Optional[str] = Field(None, description="ID of the detect intent response.")

class AsyncUpdatePayload(BaseModel):
    creditId: str = Field(..., description="The ID of the credit related to the update.")
    userId: str = Field(..., description="The ID of the user associated with the update.")
    status: str = Field(..., description="The status of the update (e.g., 'analysis_complete').")
    summary: Optional[str] = Field(None, description="Summary of the update.")
    details_url: Optional[str] = Field(None, description="URL for more details.")

class StatusUpdatePayload(BaseModel):
    dispute_id: int = Field(..., description="The ID of the dispute to update.")
    status: str = Field(..., description="The new status of the dispute.")

class GenerateRequest(BaseModel):
    contents: str = Field(..., description="The content to send to the LLM.")
    model: str = Field("gemini-pro", description="The LLM model to use (e.g., 'gemini-pro').")


# --- Core Logic Functions ---
async def _send_email_via_service(recipient_email: EmailStr, subject: str, body: str, html_body: Optional[str] = None) -> Dict[str, Any]:
    """
    Makes an HTTP POST request to the external Email Service to send an email.
    """
    if not EMAIL_SERVICE_URL:
        logger.error("EMAIL_SERVICE_URL is not configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email service URL is not configured. Cannot send email."
        )

    payload = {
        "recipient_email": recipient_email,
        "subject": subject,
        "body": body,
        "html_body": html_body
    }
    logger.info(f"Attempting to send email via Email Service to: {recipient_email}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{EMAIL_SERVICE_URL}/send_email",
                json=payload,
                timeout=30.0 # Timeout for email sending request
            )
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            logger.info(f"Email Service response: {response.json()}")
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Network error calling Email Service: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect to email service: {e}"
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Email Service returned an error (Status: {e.response.status_code}): {e.response.text}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Email service error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Unexpected error when calling Email Service: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred during email request: {e}")


async def _handle_dialogflow_fulfillment_logic(dialogflow_request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core logic to process Dialogflow CX webhook requests and formulate a response.
    This function routes requests to other services based on detected intent.
    """
    intent_display_name = dialogflow_request_data.get("fulfillmentInfo", {}).get("tag", "Unknown Intent")
    fulfillment_text = ""
    logger.info(f"Processing Dialogflow intent: '{intent_display_name}'")

    if intent_display_name == "Welcome":
        fulfillment_text = "Hello! How can I assist you today?"
    elif intent_display_name == "Fallback":
        fulfillment_text = "I'm sorry, I didn't understand that. Can you please rephrase?"
    elif intent_display_name == "TriggerCreditDisputeAnalysis":
        try:
            session_parameters = dialogflow_request_data.get("sessionInfo", {}).get("parameters", {})
            user_id = session_parameters.get("userId")
            report_id = session_parameters.get("reportId")

            if user_id is None or report_id is None:
                logger.warning(f"Missing userId or reportId for 'TriggerCreditDisputeAnalysis'. User: {user_id}, Report: {report_id}")
                return {"fulfillmentResponse": {"messages": [{"text": {"text": ["Please provide both a user ID and a report ID to start the analysis."]}}]}}

            try:
                report_id_int = int(report_id)
            except ValueError:
                logger.error(f"Invalid reportId received: '{report_id}'. Must be an integer.")
                return {"fulfillmentResponse": {"messages": [{"text": {"text": ["The report ID provided is invalid. Please ensure it's a number."]}}]}}

            analysis_request_payload = {"user_id": user_id, "report_id": report_id_int}
            
            logger.info(f"Calling Credit Dispute Service at {CREDIT_DISPUTE_SERVICE_URL}/analyze_and_dispute for user {user_id}, report {report_id_int}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{CREDIT_DISPUTE_SERVICE_URL}/analyze_and_dispute",
                    json=analysis_request_payload,
                    timeout=60.0
                )
                response.raise_for_status()
                analysis_result = response.json()
                
                identified_errors_count = len(analysis_result.get('identified_errors', []))
                letter_generated_status = "generated" if analysis_result.get('dispute_letter_content') and "No dispute letter" not in analysis_result['dispute_letter_content'] else "not generated"
                
                fulfillment_text = (
                    f"Credit analysis initiated for user {user_id}, report {report_id_int}. "
                    f"Identified {identified_errors_count} potential errors. "
                    f"Dispute letter: {letter_generated_status}."
                )
                logger.info(f"Credit Dispute Service responded successfully. Fulfillment text: {fulfillment_text}")

        except httpx.RequestError as e:
            logger.error(f"Error calling Credit Dispute Service at {CREDIT_DISPUTE_SERVICE_URL}: {e}", exc_info=True)
            fulfillment_text = "I'm sorry, I couldn't initiate the credit analysis due to a service communication issue."
        except Exception as e:
            logger.error(f"Unexpected error during credit analysis trigger from Dialogflow: {e}", exc_info=True)
            fulfillment_text = "An unexpected error occurred while processing your request."
    elif intent_display_name == "SendEmail": # New: Intent to send an email
        try:
            session_parameters = dialogflow_request_data.get("sessionInfo", {}).get("parameters", {})
            recipient_email = session_parameters.get("recipientEmail")
            subject = session_parameters.get("emailSubject", "Automated Email from Mooney AI")
            body = session_parameters.get("emailBody", "This is an automated email from Mooney AI.")

            if not recipient_email:
                logger.warning("Missing recipientEmail for 'SendEmail' intent.")
                return {"fulfillmentResponse": {"messages": [{"text": {"text": ["To send an email, please provide the recipient's email address."]}}]}}
            
            # Call the new email service
            email_response = await _send_email_via_service(
                recipient_email=recipient_email,
                subject=subject,
                body=body
            )
            fulfillment_text = f"Email successfully sent to {recipient_email}. Status: {email_response.get('message', 'Unknown')}"
            logger.info(f"Email sent via service: {fulfillment_text}")

        except HTTPException as e:
            logger.error(f"Error sending email via service: {e.detail}", exc_info=True)
            fulfillment_text = f"I encountered an error trying to send the email: {e.detail}"
        except Exception as e:
            logger.error(f"Unexpected error during email sending intent: {e}", exc_info=True)
            fulfillment_text = "An unexpected error occurred while processing your email request."
    else:
        fulfillment_text = f"Acknowledged intent: '{intent_display_name}'. This is a placeholder response for further routing."
        logger.info(f"No specific routing logic for intent: {intent_display_name}. Returning placeholder.")

    response_payload = {
        "fulfillmentResponse": {
            "messages": [
                {
                    "text": {
                        "text": [fulfillment_text]
                    }
                }
            ]
        }
    }
    return response_payload

async def _process_async_update_logic(update_data: AsyncUpdatePayload) -> Dict[str, str]:
    """
    Core logic to process asynchronous updates from other services.
    """
    logger.info(f"Received asynchronous update for Credit ID: {update_data.creditId}, User ID: {update_data.userId}, Status: {update_data.status}")
    if update_data.status == "analysis_complete":
        logger.info(f"Credit analysis complete for Credit ID: {update_data.creditId}, User ID: {update_data.userId}")
        if update_data.summary:
            logger.info(f"Summary: {update_data.summary}")
        if update_data.details_url:
            logger.info(f"Details URL: {update_data.details_url}")
        return {"status": "processed", "message": "Analysis completion update processed"}
    else:
        logger.warning(f"Unrecognized status received in async update: {update_data.status}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"status": "error", "message": f"Unrecognized status: {update_data.status}"}
        )

async def _send_updates_to_api_gateway(updates: list) -> None:
    """
    Sends updates (e.g., user notifications) to the API Gateway's notification endpoint.
    """
    full_url = f"{API_GATEWAY_URL}{API_GATEWAY_NOTIFICATION_ENDPOINT}"
    logger.info(f"Attempting to send {len(updates)} updates to API Gateway at {full_url}")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(full_url, json=updates, timeout=10)
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
            logger.info(f"Successfully sent {len(updates)} updates to API Gateway. Response status: {response.status_code}")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error sending updates to API Gateway (Status: {e.response.status_code}): {e.response.text}", exc_info=True)
        raise # Re-raise for API Gateway communication errors to be handled upstream
    except httpx.RequestError as e:
        logger.error(f"Network error sending updates to API Gateway: {e}", exc_info=True)
        raise # Re-raise for network errors


# --- LLM Integration for /api/generate endpoint ---
def _get_gemini_llm(model_name: str = "gemini-pro") -> ChatGoogleGenerativeAI:
    """Initializes and returns a ChatGoogleGenerativeAI instance."""
    if GOOGLE_API_KEY == "TODO_YOUR_GEMINI_API_KEY" or not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY is not set. Please configure it.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_API_KEY is not set. Get an API key at https://g.co/ai/idxGetGeminiKey and configure it."
        )
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Ensure env var is set for LangChain
    return ChatGoogleGenerativeAI(model=model_name)

async def _stream_gemini_response(content: str, model_name: str) -> AsyncGenerator[str, None]:
    """
    Asynchronously streams responses from the Gemini API.
    """
    try:
        model = _get_gemini_llm(model_name)
        message = HumanMessage(content=content)
        async for chunk in model.astream([message]):
            yield f'data: {json.dumps({"text": chunk.content})}\n\n'
    except HTTPException as e:
        logger.error(f"HTTPException during Gemini streaming: {e.detail}", exc_info=True)
        yield f'data: {json.dumps({"error": e.detail})}\n\n'
    except Exception as e:
        logger.error(f"Unexpected error during Gemini streaming: {e}", exc_info=True)
        yield f'data: {json.dumps({"error": str(e)})}\n\n'


# --- API Endpoints for Mooney AI Agent Service ---

# Serve static files from the 'web' directory at the root URL
app.mount("/", StaticFiles(directory="web", html=True), name="static")


@app.get("/health", response_model=Dict[str, str])
async def get_health_status() -> Dict[str, str]:
    """
    Endpoint to check the basic health of the Mooney AI Agent Service.
    """
    logger.info("Health check requested.")
    return {"message": "Mooney AI Agent Service is running"}

@app.get("/status", response_model=Dict[str, str])
async def get_service_status() -> Dict[str, str]:
    """
    Endpoint to check the operational status of the Agent Service.
    Includes database pool status.
    """
    logger.info("Status check requested.")
    db_status = "connected" if db_pool and not db_pool.closed else "disconnected/uninitialized"
    return {"status": "running", "message": "Mooney AI Agent Service is operational", "database_pool_status": db_status}

@app.post("/test_intent", response_model=Dict[str, Any])
async def test_intent_endpoint(request_body: DialogflowRequest) -> Dict[str, Any]:
    """
    Endpoint to test intent fulfillment logic directly.
    Accepts a JSON request body mimicking a Dialogflow request.
    """
    logger.info("Test intent endpoint called.")
    response = await _handle_dialogflow_fulfillment_logic(request_body.model_dump(by_alias=True))
    return response

@app.post("/webhook", response_model=Dict[str, Any]) # Response model for Dialogflow webhook
async def webhook_endpoint(request: Request) -> JSONResponse:
    """
    Webhook endpoint for receiving requests from the conversational platform.
    Handles Dialogflow CX webhook requests.
    """
    try:
        dialogflow_request_json = await request.json()
        dialogflow_request = DialogflowRequest(**dialogflow_request_json) # Validate incoming request
        
        logger.info(f"Received Dialogflow CX webhook. Session: {dialogflow_request.sessionInfo.get('session')}, Intent Tag: {dialogflow_request.fulfillmentInfo.get('tag')}")
        
        response_payload = await _handle_dialogflow_fulfillment_logic(dialogflow_request.model_dump(by_alias=True))
        return JSONResponse(content=response_payload, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error processing webhook request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"fulfillmentText": "An error occurred during webhook processing."}
        )

@app.post("/async_update", response_model=Dict[str, str])
async def async_update_endpoint(update_payload: AsyncUpdatePayload) -> Dict[str, str]:
    """
    Endpoint to receive asynchronous updates from other services.
    """
    logger.info("Received /async_update endpoint call.")
    try:
        response_data = await _process_async_update_logic(update_payload)
        return response_data
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions directly
    except Exception as e:
        logger.error(f"Unexpected error processing async update: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"status": "error", "message": f"Internal server error: {e}"}
        )

@app.post("/update_status", response_model=Dict[str, str])
async def update_status_endpoint(payload: StatusUpdatePayload) -> Dict[str, str]:
    """
    Receives updates on dispute statuses from other services and updates the database.
    This endpoint serves as the notification receiver for the Credit Dispute Service.
    """
    logger.info(f"Received /update_status endpoint call for dispute {payload.dispute_id}: {payload.status}")
    try:
        if db_pool is None:
            logger.error("Database pool not initialized when /update_status was called.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"message": "Database service is not available."}
            )
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE disputes SET status = $1, updated_at = $2 WHERE id = $3",
                payload.status,
                datetime.now(timezone.utc),
                payload.dispute_id
            )
        logger.info(f"Database: Status for dispute {payload.dispute_id} updated to {payload.status}.")
        return {"message": f"Status for dispute {payload.dispute_id} updated to {payload.status}"}
    except Exception as e:
        logger.error(f"Error updating status for dispute {payload.dispute_id} in DB: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"Error updating status for dispute {payload.dispute_id}: {e}"}
        )

@app.get("/generate_updates", response_model=Dict[str, Any])
async def generate_updates_endpoint() -> Dict[str, Any]:
    """
    Generates periodic updates for users with unresolved disputes and sends them to the API Gateway.
    """
    logger.info("Received /generate_updates endpoint call. Generating user updates...")
    try:
        if db_pool is None:
            logger.error("Database pool not initialized when /generate_updates was called.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"message": "Database service is not available."}
            )
        async with db_pool.acquire() as conn:
            disputes = await conn.fetch(
                """
                SELECT d.id, d.user_id, d.status, d.updated_at, u.username
                FROM disputes d
                JOIN users u ON d.user_id = u.id
                WHERE d.status != 'resolved'
                ORDER BY d.updated_at DESC
                """
            )

        update_messages = []
        for dispute in disputes:
            message = (
                f"Dear {dispute['username']}, there is an update regarding your credit dispute "
                f"(Reference ID: {dispute['id']}). The current status is: '{dispute['status']}'. "
                f"This was last updated on {dispute['updated_at'].strftime('%Y-%m-%d at %H:%M:%S')}."
            )
            update_messages.append({
                "dispute_id": dispute['id'],
                "user_id": dispute['user_id'],
                "message": message
            })

        if update_messages:
            try:
                await _send_updates_to_api_gateway(update_messages)
                logger.info(f"Successfully sent {len(update_messages)} updates to API Gateway.")
            except HTTPException as e: # Catch HTTPException from _send_updates_to_api_gateway
                raise e # Re-raise to ensure proper FastAPI error handling
            except Exception as e:
                logger.error(f"Unexpected error during API Gateway notification for generated updates: {e}", exc_info=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to notify API Gateway: {e}")
        else:
            logger.info("No unresolved disputes found to generate updates for.")

        return {"message": "Successfully generated user updates.", "updates": update_messages}

    except HTTPException as e:
        raise e # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Unexpected error generating updates: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": f"An internal error occurred while generating updates: {e}"}
        )

@app.post("/api/generate", response_model=StreamingResponse)
async def generate_api_endpoint(request_body: GenerateRequest) -> StreamingResponse:
    """
    Endpoint to generate text using Google Gemini and stream the response.
    """
    logger.info(f"Received /api/generate request for model: {request_body.model}")
    return StreamingResponse(_stream_gemini_response(request_body.contents, request_body.model), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    # Set dummy environment variables for local testing
    os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")
    os.environ.setdefault("GOOGLE_API_KEY", "TODO_YOUR_GEMINI_API_KEY")
    os.environ.setdefault("CREDIT_DISPUTE_SERVICE_URL", "http://localhost:8002")
    # API Gateway URL and Notification Endpoint are configured for local self-notification
    os.environ.setdefault("API_GATEWAY_URL", "http://localhost:8004")
    os.environ.setdefault("API_GATEWAY_NOTIFICATION_ENDPOINT", "/update_status") # Mooney AI Agent Service's own status update endpoint
    os.environ.setdefault("EMAIL_SERVICE_URL", "http://localhost:8000") # Points to the new Email Service

    uvicorn.run(app, host="0.0.0.0", port=8004) # Mooney AI Agent Service runs on port 8004
