import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os
from dotenv import load_dotenv
from sqlmodel import create_engine, Session, select
from contextlib import contextmanager

from typing import Optional
from sqlmodel import Field, SQLModel

class Feedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    comment: Optional[str] = None
    verified_sentiment: Optional[str] = Field(default=None, index=True)
    # Add other fields only if you need them, but this is the minimum
    resource_id: Optional[str] = Field(default=None)
    rating: Optional[int] = Field(default=None)
    sentiment: Optional[str] = Field(default=None)
    created_at: Optional[str] = Field(default=None)

# --- DATABASE CONNECTION ---
print("Loading database configuration...")
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("Error: DATABASE_URL not found in .env file.")
    print("Please ensure your .env file is in the root folder.")
    exit()

engine = create_engine(DATABASE_URL)

@contextmanager
def get_session():
    """Provides a database session."""
    with Session(engine) as session:
        yield session

def load_data_from_db():
    """
    Fetches the human-verified data from the PostgreSQL database.
    """
    print("Connecting to database to fetch verified data...")
    with get_session() as session:
        # Create a statement to select only rows
        # where 'verified_sentiment' is NOT NULL
        statement = select(Feedback.comment, Feedback.verified_sentiment).where(
            Feedback.verified_sentiment != None
        )
        
        results = session.exec(statement).all()
        
        if not results:
            print("No verified data found in database. Aborting.")
            return None
        
        print(f"Successfully fetched {len(results)} verified data entries.")
        
        # Convert the list of (comment, sentiment) tuples into a DataFrame
        df = pd.DataFrame(results, columns=['review', 'sentiment'])
        return df


# --- MAIN TRAINING LOGIC --

print("Starting model training process...")

# Load Data
df = load_data_from_db()

if df is not None:
    # 2. Define Features (X) and Labels (y)
    X = df['review']
    y = df['sentiment']

    # Check if we have enough data to split
    if len(df) < 10:
        print("Not enough data to create a test split. Need at least 10 entries.")
        print("Training on 100% of the data instead.")
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        # 3. Split Data into Training and Testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 4. Create an ML Pipeline
    print("Building ML pipeline...")
    ml_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', LogisticRegression(solver='liblinear'))
    ])

    # 5. Train the Model
    print("Training the model...")
    ml_pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 6. Evaluate the Model (if we have a test set)
    if X_test is not None:
        print("\nEvaluating model performance on test data:")
        y_pred = ml_pipeline.predict(X_test)
        print(classification_report(y_test, y_pred, zero_division=0))

    # 7. Save the NEW Trained Model
    model_filename = 'sentiment_model_v2.pkl'
    joblib.dump(ml_pipeline, model_filename)

    print(f"\nSUCCESS: New model has been trained and saved as '{model_filename}'")
else:
    print("Could not train model as no data was loaded.")
