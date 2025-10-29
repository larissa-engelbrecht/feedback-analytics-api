import os
from typing import Optional
from fastapi import FastAPI, Depends
from sqlmodel import Field, SQLModel, create_engine, Session, select
from dotenv import load_dotenv
from textblob import TextBlob   # For sentiment analysis

# Load environment variables from .env file
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

#Create the engine.
engine = create_engine(DATABASE_URL, echo=True)

# Define the Feedback model
class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: Optional[str] = Field(default_factory=str, sa_column_kwargs={"default": "NOW()"}) # Using str for simplicity, can refine to datetime
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
    create_db_and_tables()

# Define API endpoints

@app.post("/feedback/", response_model=Feedback)
def create_feedback(feedback: Feedback, session: Session = Depends(get_session)):
    """
    Create a new feedback entry.
    """
    # We don't run sentiment analysis yet, just save the feedback

    if feedback.comment:
        # Create a TextBlob object
        blob = TextBlob(feedback.comment)

        # Get the polarity score (-1 = very negative, 1 = very positive)
        sentiment_score = blob.sentiment.polarity
        if sentiment_score > 0.1:
            feedback.sentiment = "positive"
        elif sentiment_score < -0.1:
            feedback.sentiment = "negative"
        else:
            feedback.sentiment = "neutral"

    session.add(feedback)
    session.commit()
    session.refresh(feedback)
    return feedback

@app.get("/feedback/", response_model=list[Feedback])
def read_feedbacks(session: Session = Depends(get_session)):
    """
    Retrieve all feedback entries.
    """
    feedbacks = session.exec(select(Feedback)).all()
    return feedbacks