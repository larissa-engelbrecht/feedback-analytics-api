import os
from typing import Optional
from fastapi import FastAPI, Depends
from sqlmodel import Field, SQLModel, create_engine, Session, select
from dotenv import load_dotenv
import joblib
from sklearn.pipeline import Pipeline
from datetime import datetime
from sqlalchemy import Column, DateTime


# Load environment variables from .env file
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

#Create the engine.
engine = create_engine(DATABASE_URL, echo=True)

model: Optional[Pipeline] = None
MODEL_PATH = "sentiment_model_v1.pkl"

# Define the Feedback model
class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), server_default="NOW()")
    ) # Using str for simplicity, can refine to datetime
    resource_id: str = Field(index=True) 
    rating: int = Field(ge=1, le=10) # Rating between 1 and 10 - "ge" means greater than or equal to, "le" means less than or equal to
    comment: Optional[str] = None
    sentiment: Optional[str] = None
    # Todo: Add user_context field later

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

# Define API endpoints

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
    return feedback

@app.get("/feedback/", response_model=list[Feedback])
def read_feedbacks(session: Session = Depends(get_session)):
    """
    Retrieve all feedback entries.
    """
    feedback_list = session.exec(select(Feedback)).all()
    return feedback_list