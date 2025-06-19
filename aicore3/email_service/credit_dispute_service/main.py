import os
import json
import httpx
import asyncpg
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# LangChain and LLM imports for Gemini integration
from langchain_core.messages import HumanMessage, SystemMessage # SystemMessage added for LangGraph prompts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

app = FastAPI(
    title="Credit Dispute Service",
    description="Dedicated service for credit report analysis, error identification, and dispute letter generation.",
    version="1.0.0"
)

# --- Configuration Management ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "TODO_YOUR_GEMINI_API_KEY") # Google Gemini API Key
# URL for the Progress Tracking Service (Mooney AI Agent Service's /update_status endpoint)
PROGRESS_TRACKING_SERVICE_URL = os.getenv("PROGRESS_TRACKING_SERVICE_URL", "http://localhost:8004/update_status")


# --- Global Resources ---
db_pool: Optional[asyncpg.Pool] = None

# --- Pydantic Models ---
class ErrorDetail(BaseModel):
    """Represents a single identified error in the credit report."""
    reason: str = Field(..., description="The reason why this is considered an error, referencing relevant laws or common credit reporting standards if applicable.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="A confidence score (0.0 to 1.0) indicating the likelihood of this being a genuine error.")

class AnalysisState(BaseModel):
    """
    Represents the state of the credit analysis workflow in LangGraph.
    This model holds all data needed as the workflow progresses.
    """
    user_id: int
    report_id: int # This report_id is the SERIAL PRIMARY KEY 'id' from the credit_reports table
    report_data: Optional[Dict[str, Any]] = None # Structured data from the credit report
    user_data: Optional[Dict[str, Any]] = None   # User's personal information
    identified_errors: List[ErrorDetail] = Field([], description="List of identified errors.")
    dispute_letter_content: Optional[str] = Field(None, description="The generated dispute letter text.")

class AnalysisRequest(BaseModel):
    """Request model for initiating credit report analysis."""
    user_id: int = Field(..., description="The ID of the user requesting the analysis.")
    report_id: int = Field(..., description="The internal database ID of the credit report to be analyzed.")

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event() -> None:
    """Initializes database connection pool and loads AI models on application startup."""
    global db_pool
    print("Credit Dispute Service: Starting up...") # Use print for startup messages before logger is fully configured
    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("Credit Dispute Service: Database connection pool created successfully.")
    except Exception as e:
        print(f"Credit Dispute Service: Failed to create database connection pool: {e}")
        raise # Critical error, prevent app from starting

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Closes database connection pool on application shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()
        print("Credit Dispute Service: Database connection pool closed.")
    print("Credit Dispute Service: Shut down complete.")

# --- Database Operations Layer ---
async def _fetch_report_and_user_data(user_id: int, report_id: int) -> Dict[str, Any]:
    """
    Fetches structured credit report data and user information from the database.
    """
    if db_pool is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database pool not initialized.")

    async with db_pool.acquire() as conn:
        # Fetch report using its internal database ID (integer primary key)
        report_row = await conn.fetchrow(
            "SELECT structured_data FROM credit_reports WHERE id = $1 AND user_id = $2",
            report_id, user_id
        )
        if not report_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credit report not found for this user with the given ID.")

        user_row = await conn.fetchrow("SELECT id, username, email FROM users WHERE id = $1", user_id)
        if not user_row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

        return {
            "report_data": report_row['structured_data'],
            "user_data": dict(user_row) # Convert record to dict for Pydantic model compatibility
        }

async def _store_analysis_results(
    user_id: int,
    report_id: int,
    identified_errors: List[ErrorDetail],
    generated_letter: Optional[str]
) -> int:
    """
    Stores the analysis results (identified errors) and the generated dispute letter in the database.
    """
    if db_pool is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database pool not initialized.")

    async with db_pool.acquire() as conn:
        dispute_id = await conn.fetchval(
            """
            INSERT INTO disputes (user_id, credit_report_id, status, identified_errors, generated_letter)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (user_id, credit_report_id) DO UPDATE
            SET status = EXCLUDED.status, identified_errors = EXCLUDED.identified_errors,
                generated_letter = EXCLUDED.generated_letter, updated_at = NOW()
            RETURNING id;
            """,
            user_id,
            report_id,
            "analysis_complete", # Initial status for the dispute record
            json.dumps([e.model_dump() for e in identified_errors]), # Convert list of Pydantic models to JSON string
            generated_letter
        )
        print(f"Credit Dispute Service: Stored analysis results and letter for user {user_id}, report {report_id}. Dispute ID: {dispute_id}")
        return dispute_id

# --- LLM Integration Layer ---
def _get_gemini_llm(model_name: str = "gemini-pro") -> ChatGoogleGenerativeAI:
    """Initializes and returns a ChatGoogleGenerativeAI instance."""
    if GOOGLE_API_KEY == "TODO_YOUR_GEMINI_API_KEY" or not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_API_KEY is not set for Credit Dispute Service. Get an API key at https://g.co/ai/idxGetGeminiKey"
        )
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY # Ensure environment variable is set for langchain
    return ChatGoogleGenerativeAI(model=model_name)

async def _call_llm_api(prompt_messages: List[Any], parser: Any = None, model_name: str = "gemini-pro") -> Any:
    """
    Calls the configured LLM API with the given prompt.
    This function is for non-streaming internal LangGraph calls.
    """
    try:
        llm_client = _get_gemini_llm(model_name)

        # LangChain's ainvoke expects a List[BaseMessage] or str
        if isinstance(prompt_messages, list) and all(isinstance(msg, (HumanMessage, SystemMessage)) for msg in prompt_messages):
            raw_response = await llm_client.ainvoke(prompt_messages)
        elif isinstance(prompt_messages, str):
            raw_response = await llm_client.ainvoke([HumanMessage(content=prompt_messages)])
        else:
            # Attempt to convert to HumanMessage if it's a list of strings/dicts not yet BaseMessage
            processed_messages = []
            for msg_content in prompt_messages:
                if isinstance(msg_content, (str, dict)): # Assuming dicts might be serialized directly
                    processed_messages.append(HumanMessage(content=str(msg_content)))
                else:
                    processed_messages.append(msg_content) # Keep as is if already BaseMessage
            raw_response = await llm_client.ainvoke(processed_messages)


        if parser:
            # Parse the content of the AI message response
            return parser.parse(raw_response.content)
        return raw_response.content

    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e:
        print(f"Credit Dispute Service: Error calling LLM API: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM API call failed: {e}")

# --- RAG Logic Layer ---
def _generate_query_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for the given text.
    In a real system, this would use an embedding model. Placeholder for now.
    """
    # This should ideally call an external embedding service or use a loaded model
    # For now, it's a dummy placeholder. In ingestion service, SentenceTransformer is used.
    # To be consistent, you might load SentenceTransformer here too, or call your Ingestion Service's embedding API.
    return [0.1] * 384 # Matches 'all-MiniLM-L6-v2' dimension

async def _get_relevant_rag_context(query_text: str, current_report_id: int) -> Dict[str, Any]:
    """
    Performs Retrieval-Augmented Generation (RAG) to fetch relevant context
    from the vector database (credit_reports and knowledge_base tables).
    """
    if db_pool is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database pool not initialized.")

    query_embedding = _generate_query_embedding(query_text)
    async with db_pool.acquire() as conn:
        # Fetch other relevant credit report chunks (excluding the current one)
        relevant_report_chunks = await conn.fetch(
            """
            SELECT structured_data FROM credit_reports
            WHERE id != $1
            ORDER BY embedding <-> $2 LIMIT 5; -- Using L2 distance for vector search
            """,
            current_report_id, query_embedding
        )
        # Fetch relevant knowledge base entries
        relevant_knowledge = await conn.fetch(
            """
            SELECT content FROM knowledge_base
            ORDER BY embedding <-> $1 LIMIT 5; -- Using L2 distance for vector search
            """,
            query_embedding
        )

        # Extract content (assuming structured_data is JSONB and content is TEXT)
        report_chunks_content = [row['structured_data'] for row in relevant_report_chunks]
        knowledge_content = [row['content'] for row in relevant_knowledge]

        return {"report_chunks": report_chunks_content, "knowledge": knowledge_content}

def _format_rag_context_for_llm(rag_results: dict) -> str:
    """
    Formats the results from RAG queries into a coherent string
    that can be used as context for the LLM.
    """
    formatted_context = ""
    if rag_results.get("report_chunks"):
        formatted_context += "--- Relevant Credit Report Chunks from other reports ---\n" + \
                             "\n---\n".join([json.dumps(c, indent=2) for c in rag_results["report_chunks"]]) + "\n\n"
    if rag_results.get("knowledge"):
        formatted_context += "--- Relevant Knowledge Base Information ---\n" + \
                             "\n---\n".join(rag_results["knowledge"])
    return formatted_context

# --- LangGraph Workflow Node Functions ---

async def _node_validate_input_data(state: AnalysisState) -> AnalysisState:
    """Node: Validates the initial input data for the analysis process."""
    print("Credit Dispute Service Workflow Node: Input data validation complete.")
    # Fetch report data and user data as part of initial validation/enrichment
    fetched_data = await _fetch_report_and_user_data(state.user_id, state.report_id)
    state.report_data = fetched_data["report_data"]
    state.user_data = fetched_data["user_data"]
    return state

async def _node_identify_errors(state: AnalysisState) -> AnalysisState:
    """Node: Identifies potential errors in the credit report data using the LLM and RAG."""
    print("Credit Dispute Service Workflow Node: Identifying errors in credit report...")

    # Query RAG for relevant context
    rag_context = await _get_relevant_rag_context(
        f"Identify potential errors and inconsistencies in credit report for user {state.user_id}",
        state.report_id
    )
    formatted_context = _format_rag_context_for_llm(rag_context)

    # Define the prompt for LLM to identify errors
    error_identification_prompt_template = PromptTemplate.from_messages([
        SystemMessage(
            "You are an expert credit analyst. Analyze the provided credit report data "
            "and identify any potential errors, inconsistencies, or questionable entries. "
            "Reference relevant laws or common credit reporting standards from the provided information."
            "The output MUST be a JSON array of error objects, strictly following the specified schema."
        ),
        HumanMessage(
            content=(
                "<credit_report_data>\n{credit_report_data}\n</credit_report_data>\n\n"
                "Based on this data, identify any information that appears incorrect, contradictory, "
                "or requires further investigation.\n\n"
                "Provide relevant information from the knowledge base and other credit report chunks to support your findings:\n"
                "<relevant_information>\n{relevant_information}\n</relevant_information>\n\n"
                "Your response should be a JSON array of error objects. Each error object must have the following keys:\n"
                "- `reason`: The reason why this is considered an error, referencing relevant laws or common credit reporting standards if applicable. Be specific (e.g., 'Account reported as open, but it was closed onAPAC-MM-DD. This violates [Relevant Law]'). If referencing a law, mention its name or key principle if available in the context.\n"
                "- `confidence`: A float number between 0.0 (very low confidence) and 1.0 (very high confidence) indicating your confidence level that this is a genuine error that can be disputed. If you are unsure, use a lower confidence score."
            )
        )
    ])

    try:
        # Prepare messages for the LLM
        messages_for_llm = [
            SystemMessage(content=error_identification_prompt_template.messages[0].content), # System prompt
            HumanMessage(content=error_identification_prompt_template.messages[1].content.format(
                credit_report_data=json.dumps(state.report_data, indent=2),
                relevant_information=formatted_context
            ))
        ]

        llm_response = await _call_llm_api(
            prompt_messages=messages_for_llm,
            parser=JsonOutputParser() # Expecting JSON output
        )

        identified_errors = []
        if isinstance(llm_response, list): # Ensure LLM response is a list
            for error_data in llm_response:
                try:
                    error_detail = ErrorDetail(**error_data)
                    identified_errors.append(error_detail)
                except ValidationError as e:
                    print(f"Credit Dispute Service Warning: LLM response element failed Pydantic validation: {error_data}. Error: {e}")
                except Exception as e:
                    print(f"Credit Dispute Service Warning: Failed to parse an identified error from LLM response: {error_data}. Error: {e}")
        else:
            print(f"Credit Dispute Service Warning: LLM response was not a list as expected: {llm_response}")

        state.identified_errors = identified_errors
        print(f"Credit Dispute Service Workflow Node: Identified {len(identified_errors)} potential errors.")
        return state

    except Exception as e:
        print(f"Credit Dispute Service Error during error identification: {e}")
        raise # Re-raise to halt workflow if critical

async def _node_generate_dispute_letter(state: AnalysisState) -> AnalysisState:
    """Node: Generates a formal dispute letter if errors were identified."""
    print("Credit Dispute Service Workflow Node: Generating dispute letter...")

    if not state.identified_errors:
        print("Credit Dispute Service: No errors found, skipping dispute letter generation.")
        state.dispute_letter_content = "No dispute letter generated as no errors were identified."
        return state

    # Format identified errors for the prompt
    errors_for_prompt = "\n".join([f"- Reason: {e.reason}\n  Confidence: {e.confidence:.2f}" for e in state.identified_errors])

    # Define the prompt for LLM to generate dispute letter
    dispute_letter_prompt_template = PromptTemplate.from_messages([
        SystemMessage(
            "You are a professional letter writer specializing in credit disputes. "
            "Draft a formal dispute letter to a credit bureau based on the identified errors and user information. "
            "Ensure the letter is polite, professional, and clearly states the disputed items and reasons. "
            "Do NOT include placeholders like '[Your Name]' but use actual user data provided. "
            "The letter should be ready for submission."
        ),
        HumanMessage(
            content=(
                "User Information:\n{user_info}\n\n"
                "Identified Errors:\n{identified_errors}\n\n"
                "Credit Report Data (for reference, do not copy verbatim):\n{report_data}\n\n"
                "Please draft a formal dispute letter."
            )
        )
    ])

    try:
        # Prepare messages for the LLM
        messages_for_llm = [
            SystemMessage(content=dispute_letter_prompt_template.messages[0].content),
            HumanMessage(content=dispute_letter_prompt_template.messages[1].content.format(
                user_info=json.dumps(state.user_data, indent=2),
                identified_errors=errors_for_prompt,
                report_data=json.dumps(state.report_data, indent=2)
            ))
        ]

        generated_letter = await _call_llm_api(
            prompt_messages=messages_for_llm,
            parser=StrOutputParser() # Expecting string output
        )
        state.dispute_letter_content = generated_letter
        print("Credit Dispute Service Workflow Node: Dispute letter generated.")
        return state
    except Exception as e:
        print(f"Credit Dispute Service Error during dispute letter generation: {e}")
        state.dispute_letter_content = f"Failed to generate dispute letter due to an internal error: {e}"
        raise # Re-raise to halt workflow if critical

async def _node_store_results(state: AnalysisState) -> AnalysisState:
    """Node: Stores the analysis results and generated dispute letter in the database."""
    print("Credit Dispute Service Workflow Node: Storing analysis results and dispute letter...")
    try:
        # The dispute_id returned here is the database's internal SERIAL ID
        dispute_db_id = await _store_analysis_results(
            user_id=state.user_id,
            report_id=state.report_id,
            identified_errors=state.identified_errors,
            generated_letter=state.dispute_letter_content
        )
        # You might want to store this dispute_db_id in the state if needed later
        # state.dispute_db_id = dispute_db_id # Add this to AnalysisState if needed
        print(f"Credit Dispute Service Workflow Node: Results stored successfully. DB Dispute ID: {dispute_db_id}")
        return state
    except Exception as e:
        print(f"Credit Dispute Service Error storing results: {e}")
        raise # Re-raise to halt workflow if critical

async def _node_notify_progress_tracking(state: AnalysisState) -> AnalysisState:
    """Node: Notifies the Progress Tracking Service (Mooney AI Agent Service) about the completion of the analysis."""
    print("Credit Dispute Service Workflow Node: Notifying progress tracking service...")
    
    if not PROGRESS_TRACKING_SERVICE_URL:
        print("Credit Dispute Service: PROGRESS_TRACKING_SERVICE_URL not set, skipping notification.")
        return state # Do not raise error if notification URL is not set

    try:
        payload = {
            # Use a dummy dispute_id or actual if you stored it in state
            # For this context, we are updating *a* dispute's status in the agent service's DB.
            # You might need to retrieve the actual dispute_id from the database
            # if the agent service needs to update a specific pre-existing dispute record.
            # For simplicity, sending a placeholder/dummy or assuming agent service handles
            # based on user_id/report_id if no specific dispute_id is generated here.
            "dispute_id": 1, # Placeholder: Replace with actual dispute_id if generated/retrieved
            "status": "analysis_complete",
            "user_id": state.user_id,
            "report_id": state.report_id,
            "identified_errors_count": len(state.identified_errors),
            "letter_generated": bool(state.dispute_letter_content and "No dispute letter" not in state.dispute_letter_content),
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(PROGRESS_TRACKING_SERVICE_URL, json=payload, timeout=10)
            response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx responses
            print(f"Credit Dispute Service: Progress tracking service notified: {response.json()}")
        return state
    except httpx.HTTPStatusError as e:
        print(f"Credit Dispute Service Error notifying progress tracking service (HTTP Status: {e.response.status_code}): {e.response.text}")
        # Log error but don't re-raise, as the analysis itself completed successfully
    except httpx.RequestError as e:
        print(f"Credit Dispute Service Network error notifying progress tracking service: {e}")
    except Exception as e:
        print(f"Credit Dispute Service Unexpected error notifying progress tracking service: {e}")
    return state

# Conditional logic for LangGraph
def _should_generate_letter(state: AnalysisState) -> str:
    """Determines whether to proceed with dispute letter generation based on identified errors."""
    if state.identified_errors:
        print("Credit Dispute Service Decision: Errors identified, generating dispute letter.")
        return "generate_dispute_letter"
    else:
        print("Credit Dispute Service Decision: No errors identified, skipping dispute letter generation.")
        return "store_results"

# Build the LangGraph workflow
workflow = StateGraph(AnalysisState)

workflow.add_node("validate_data", _node_validate_input_data)
workflow.add_node("identify_errors", _node_identify_errors)
workflow.add_node("generate_dispute_letter", _node_generate_dispute_letter)
workflow.add_node("store_results", _node_store_results)
workflow.add_node("notify_progress", _node_notify_progress_tracking)

workflow.set_entry_point("validate_data")

workflow.add_edge("validate_data", "identify_errors")
workflow.add_conditional_edges("identify_errors", _should_generate_letter)
workflow.add_edge("generate_dispute_letter", "store_results")
workflow.add_edge("store_results", "notify_progress")
workflow.add_edge("notify_progress", END)

compiled_workflow: CompiledGraph = workflow.compile()


# --- API Endpoints for Credit Dispute Service ---
@app.get("/")
async def read_root() -> Dict[str, str]:
    """Root endpoint for the Credit Dispute Service."""
    return {"message": "Credit Dispute Service is running."}

@app.get("/status")
async def get_status() -> Dict[str, str]:
    """Endpoint to check the status of the Credit Dispute Service."""
    db_status = "connected" if db_pool and not db_pool.closed else "disconnected/uninitialized"
    return {"status": "running", "message": "Credit Dispute Service is operational", "database_pool_status": db_status}

@app.post("/analyze_and_dispute", response_model=AnalysisState)
async def analyze_and_dispute_endpoint(request: AnalysisRequest) -> AnalysisState:
    """
    Initiates the credit analysis and dispute generation process for a given user and report.
    This endpoint orchestrates the entire workflow using LangGraph.
    """
    print(f"Credit Dispute Service: Received request to analyze report {request.report_id} for user {request.user_id}")
    try:
        # Initialize the LangGraph state with user_id and report_id
        initial_state = AnalysisState(
            user_id=request.user_id,
            report_id=request.report_id,
            # report_data and user_data will be fetched in _node_validate_input_data
        )

        # Execute the LangGraph workflow
        # The .ainvoke method returns the final state of the graph
        final_state: AnalysisState = await compiled_workflow.ainvoke(initial_state)

        print("Credit Dispute Service: Credit analysis and dispute generation process completed.")
        return final_state

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPExceptions directly
    except Exception as e:
        print(f"Credit Dispute Service: An unexpected error occurred during analysis workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during the analysis and dispute generation process: {e}"
        )

if __name__ == "__main__":
    import uvicorn
    # Set dummy environment variables for local testing if not already set
    os.environ.setdefault("DATABASE_URL", "postgresql://user:password@localhost:5432/mooney_db")
    os.environ.setdefault("GOOGLE_API_KEY", "TODO_YOUR_GEMINI_API_KEY") # Replace with your actual key for testing
    os.environ.setdefault("PROGRESS_TRACKING_SERVICE_URL", "http://localhost:8004/update_status") # Points to Mooney AI Agent Service
    uvicorn.run(app, host="0.0.0.0", port=8002) # Credit Dispute Service runs on port 8002
