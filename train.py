import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

print("Starting model training process...")

# Load dataset
try:
    df = pd.read_csv('IMDB_Dataset.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'IMDB_Dataset.csv' not found")
    print("Please download it from Kaggle(https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download) and place it in the current directory.")
    exit()

# Define Features (X) and Labels (y)
X = df['review']
y = df['sentiment']

# Check data balance
print("Checking data balance:")
print(df['sentiment'].value_counts(normalize=True))


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")

# Create an ML Pipeline
print("Building ML pipeline...")
ml_pipeline = Pipeline([
    # Step 1: 'vectorizer' - Turn text into a matrix of TF-IDF numbers
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),
    # Step 2: 'classifier' - The machine learning algorithm
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Train the model
print("Training the model... (This may take a few minutes)")
ml_pipeline.fit(X_train, y_train)
print("Model training complete")

# Evaluate the model
print("Evaluating model performance on test data:")
y_pred = ml_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the trained model to a file
model_filename = 'sentiment_model_v1.pkl'
joblib.dump(ml_pipeline, model_filename)

print(f"\nSUCCESS: The Model has been trained and saved as '{model_filename}'")
