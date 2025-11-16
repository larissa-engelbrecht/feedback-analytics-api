import os
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException
from sqlmodel import Field, SQLModel, create_engine, Session, select, func
from dotenv import load_dotenv
import joblib
from sklearn.pipeline import Pipeline
from datetime import datetime
from sqlalchemy import Column, DateTime
import google.generativeai as genai
import requests


# Load environment variables from .env file
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Configure the Gemini AI model
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # List all available models
    # for m in genai.list_models():
    #     print(m) 

#Create the engine.
engine = create_engine(DATABASE_URL, echo=True)

model: Optional[Pipeline] = None
MODEL_PATH = "sentiment_model_v1.pkl"


# --- MODEL DEFINITIONS ---

class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), server_default="NOW()")
    ) # Using str for simplicity, can refine to datetime
    resource_id: str = Field(index=True) 
    rating: int = Field(ge=1, le=10) # Rating between 1 and 10 - "ge" means greater than or equal to, "le" means less than or equal to
    comment: Optional[str] = None
    sentiment: Optional[str] = None
    verified_sentiment: Optional[str] = Field(default=None, index=True)

class FeedbackStats(SQLModel):
    resource_id: str
    average_rating: float
    feedback_count: int
    negative_comment_count: int
    positive_comment_count: int

class SummarizeRequest(SQLModel):
    resource_id: str

class SummarizeResponse(SQLModel):
    resource_id: str
    summary: str
    error: Optional[str] = None

class ValidateSentimentRequest(SQLModel):
    verified_sentiment: str

class WebhookSubscription(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), server_default="NOW()")
    )
    
    target_url: str = Field(index=True) # The URL to send the alert to (e.g., Zapier, Slack)
    event_type: str = Field(index=True) # e.g., "negative_only" or "all_feedback"
    
    # Optional: to scope a subscription to just one resource
    resource_id: Optional[str] = Field(default=None, index=True)

# Create the database tables
# A function to create the database and tables when the app starts
def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# Dependency to get a session
# Standard way to get a session in FastAPI with SQLModel
def get_session():
    with Session(engine) as session:
        yield session

# Initialize FastAPI app
app = FastAPI(
    title="Feedback Analytics API",
    description="A plug-and-play API for collecting and analyzing user feedback on resources.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    global model
    print("Application Startup...")

    create_db_and_tables()

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        model = None
    except Exception as e:
        print(f"ERROR: Could not load model. {e}")
        model = None


# --- DEFINE API ENDPOINTS ---

@app.post("/feedback/", response_model=Feedback)
def create_feedback(feedback: Feedback, session: Session = Depends(get_session)):
    """
    Create a new feedback entry.
    """
    # We don't run sentiment analysis yet, just save the feedback

    if feedback.comment and model:
        
        comment_list = [feedback.comment]

        # Make a prediction
        prediction = model.predict(comment_list)

        # The prediction is an array, get the first element only
        feedback.sentiment = prediction[0]

    elif feedback.comment and not model:
        # Fallback in case the model failed to load
        feedback.sentiment = "model_not_loaded"

    session.add(feedback)
    session.commit()
    session.refresh(feedback)

    # --- WEBHOOK LOGIC ---
    print("Feedback created. Checking for webhooks...")
    
    # Find all subscriptions
    webhooks = session.exec(select(WebhookSubscription)).all()
    
    # Serialize our new feedback to JSON
    # We use .model_dump_json() for a serializable format
    feedback_json = feedback.model_dump_json()

    for hook in webhooks:
        # Check if the resource_id matches (if one was specified)
        if hook.resource_id and hook.resource_id != feedback.resource_id:
            continue 

        # Check if the event type matches
        if hook.event_type == "all_feedback":
            try:
                print(f"Firing 'all_feedback' webhook to: {hook.target_url}")
                requests.post(hook.target_url, data=feedback_json, headers={"Content-Type": "application/json"})
            except Exception as e:
                print(f"Failed to send webhook {hook.id}: {e}")
        
        elif hook.event_type == "negative_only" and feedback.sentiment == "negative":
            try:
                print(f"Firing 'negative_only' webhook to: {hook.target_url}")
                requests.post(hook.target_url, data=feedback_json, headers={"Content-Type": "application/json"})
            except Exception as e:
                print(f"Failed to send webhook {hook.id}: {e}")
    # --- END OF WEBHOOK LOGIC ---

    return feedback

@app.get("/feedback/", response_model=list[Feedback])
def read_feedback(session: Session = Depends(get_session)):
    """
    Retrieve all feedback entries.
    """
    feedback_list = session.exec(select(Feedback)).all()
    return feedback_list

@app.get("/feedback/stats/", response_model=List[FeedbackStats])
def get_feedback_stats(session: Session = Depends(get_session)):
    """
    Retrieves statistics for all resources.
    """
    # This quesry calculates the average rating, total count,
    # and count of negative sentiments for each resource_id.
    statement = (
        select(
            Feedback.resource_id,
            func.avg(Feedback.rating).label("average_rating"),
            func.count(Feedback.id).label("feedback_count"),
            func.count(Feedback.id).filter(Feedback.sentiment == "negative").label("negative_comment_count"),
            func.count(Feedback.id).filter(Feedback.sentiment == "positive").label("positive_comment_count"),
        )
        .group_by(Feedback.resource_id)
    )

    results = session.exec(statement).all()

    # Convert results to the Pydantic model
    stats_list =[
        FeedbackStats(
            resource_id=row[0],
            average_rating=row[1],
            feedback_count=row[2],
            negative_comment_count=row[3] if row[3] is not None else 0,
            positive_comment_count=row[4] if row[4] is not None else 0
        )
        for row in results
    ]

    return stats_list

@app.post("/feedback/summarize/", response_model=SummarizeResponse)
def get_feedback_summary(request: SummarizeRequest, session: Session = Depends(get_session)):
    """
    Generates an AI-powered summary for all feedback on a specific resource.
    """
    if not GEMINI_API_KEY:
        return SummarizeResponse(
            resource_id=request.resource_id,
            summary="",
            error="Gemini API key not configured on server"
        )
    
    # Get all comments for the resource
    statement = select(Feedback.comment).where(
        Feedback.resource_id == request.resource_id,
        Feedback.comment != None
    )
    comments = session.exec(statement).all()

    if not comments:
        return SummarizeResponse(
            resource_id = request.resource_id,
            summary="",
            error= "No comments found for this resource."
        )
    
    # Combine comments into a single text block
    combined_comments ="\n".join([f"- {comment}" for comment in comments])

    # Create the prompt for Gemini
    prompt = f"""
    You are a helpful product manager. Analyze the following user feedback comments for the product '{request.resource_id}'.
    Provide a 2-3 bullet point summary of the main pain points and positive themes.

    Comments:
    {combined_comments}

    Summary:
    """
    # Call the AI Model (Gemini API)
    try:
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        model_response = model.generate_content(prompt)

        return SummarizeResponse(
            resource_id = request.resource_id,
            summary = model_response.text
        )
    except Exception as e:
        return SummarizeResponse(
            resource_id=request.resource_id,
            summary="",
            error=f"Error generating summary: {e}"
        )

@app.put("/feedback/{feedback_id}/validate/", response_model=Feedback)
def validate_feedback_sentiment(feedback_id: int, request: ValidateSentimentRequest, session: Session = Depends(get_session)):
    """
    Validates a feedback entry with the verified sentiment provided by a human.
    """

    # Find the entry in the database
    feedback = session.get(Feedback, feedback_id)

    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")

    # Update the verified sentiment
    feedback.verified_sentiment = request.verified_sentiment

    # Save the changes to the database
    session.add(feedback)
    session.commit()
    session.refresh(feedback)

    return feedback

# --- WEBHOOK ENDPOINTS ---

@app.post("/webhooks/", response_model=WebhookSubscription)
def create_webhook(
    subscription: WebhookSubscription, 
    session: Session = Depends(get_session)
):
    """
    Creates a new webhook subscription.
    """
    session.add(subscription)
    session.commit()
    session.refresh(subscription)
    return subscription


@app.get("/webhooks/", response_model=List[WebhookSubscription])
def get_webhooks(session: Session = Depends(get_session)):
    """
    Retrieves all existing webhook subscriptions.
    """
    webhooks = session.exec(select(WebhookSubscription)).all()
    return webhooks


@app.delete("/webhooks/{webhook_id}", response_model=dict)
def delete_webhook(webhook_id: int, session: Session = Depends(get_session)):
    """
    Deletes a webhook subscription.
    """
    webhook = session.get(WebhookSubscription, webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    session.delete(webhook)
    session.commit()
    return {"detail": "Webhook deleted successfully"}
