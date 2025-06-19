import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from datetime import datetime, timezone # Added import for datetime and timezone

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

# Configure logging for the service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EmailService")

app = FastAPI(
    title="Email Service",
    description="Dedicated service for sending emails via SMTP.",
    version="1.0.0"
)

# --- Configuration Management ---
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # Typically 587 for TLS, 465 for SSL
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "your_email@example.com") # Replace with your email
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "your_email_password") # Replace with your email password or app-specific password
SENDER_EMAIL = os.getenv("SENDER_EMAIL", SMTP_USERNAME) # The email address to send from

# --- Pydantic Models ---
class EmailSendRequest(BaseModel):
    """Request model for sending an email."""
    recipient_email: EmailStr = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject line of the email.")
    body: str = Field(..., description="The plain text body of the email.")
    html_body: Optional[str] = Field(None, description="Optional HTML body of the email.")
    # In a real system, you might include original_email_id_context for replies etc.

class EmailSendResponse(BaseModel):
    """Response model for a successful email send operation."""
    message: str = Field(..., description="Status message for the email send operation.")
    recipient: str = Field(..., description="The email address of the recipient.")
    subject: str = Field(..., description="The subject of the sent email.")
    timestamp: str = Field(..., description="Timestamp of when the email send request was processed.")

# --- Email Sending Logic ---
async def _send_email(recipient_email: str, subject: str, body: str, html_body: Optional[str] = None) -> None:
    """
    Sends an email using SMTP.
    """
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        logger.error("Email service environment variables not fully configured.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email service not configured. Please set SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL environment variables."
        )

    msg = MIMEMultipart("alternative")
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach plain text and HTML versions
    msg.attach(MIMEText(body, 'plain'))
    if html_body:
        msg.attach(MIMEText(html_body, 'html'))

    try:
        # Use a context manager for the SMTP connection
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls() # Enable TLS encryption
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        logger.info(f"Email successfully sent to {recipient_email} with subject: '{subject}'")
    except smtplib.SMTPAuthenticationError:
        logger.error(f"Failed to send email: Authentication error. Check SMTP_USERNAME and SMTP_PASSWORD for {SENDER_EMAIL}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Email authentication failed. Check credentials.")
    except smtplib.SMTPConnectError as e:
        logger.error(f"Failed to send email: SMTP connection error to {SMTP_SERVER}:{SMTP_PORT}. Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to connect to SMTP server: {e}")
    except smtplib.SMTPException as e:
        logger.error(f"Failed to send email: SMTP error occurred. Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"SMTP error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending email to {recipient_email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred while sending email: {e}")

# --- API Endpoints ---
@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint for the Email Service."""
    return {"message": "Email Service is running."}

@app.get("/status")
async def get_status() -> Dict[str, str]:
    """Endpoint to check the status of the Email Service."""
    # This check is basic; real health check might try to connect to SMTP.
    config_status = "configured" if all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD]) else "unconfigured"
    return {"status": "running", "message": "Email Service operational", "config_status": config_status}

@app.post("/send_email", response_model=EmailSendResponse)
async def send_email_endpoint(request: EmailSendRequest) -> EmailSendResponse:
    """
    Endpoint to send an email.
    """
    logger.info(f"Received request to send email to: {request.recipient_email}")
    try:
        await _send_email(request.recipient_email, request.subject, request.body, request.html_body)
        return EmailSendResponse(
            message="Email sent successfully.",
            recipient=request.recipient_email,
            subject=request.subject,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    except HTTPException as e:
        raise e # Re-raise HTTPExceptions raised by _send_email
    except Exception as e:
        logger.error(f"An unexpected error occurred in send_email_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred.")


if __name__ == "__main__":
    import uvicorn
    # Set dummy environment variables for local testing if not already set
    # IMPORTANT: Replace with actual details or use environment variables in deployment
    os.environ.setdefault("SMTP_SERVER", "smtp.gmail.com")
    os.environ.setdefault("SMTP_PORT", "587")
    os.environ.setdefault("SMTP_USERNAME", "your_email@gmail.com") # <--- REPLACE THIS
    os.environ.setdefault("SMTP_PASSWORD", "your_app_password") # <--- REPLACE WITH APP PASSWORD (for Gmail) or actual password
    os.environ.setdefault("SENDER_EMAIL", "your_email@gmail.com") # <--- REPLACE THIS

    # For Gmail: If using a regular password, you might need to enable "Less secure app access" (deprecated).
    # It's highly recommended to use an "App password" if you have 2-Factor Authentication enabled.
    # Go to your Google Account -> Security -> App passwords.

    uvicorn.run(app, host="0.0.0.0", port=8000) # Email Service runs on port 8000
