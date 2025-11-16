import requests
import random
import time
from datetime import datetime, timedelta

API_URL = "http://localhost:8002/feedback/"

# --- Define our inputs ---

RESOURCES = [
    "doc-101-install-guide",
    "doc-102-api-ref",
    "product-xyz-feature",
    "product-abc-checkout",
    "feature-searchbar",
    "support-ticket-widget"
]

POSITIVE_COMMENTS = [
    "This is absolutely amazing! I love it.",
    "Worked perfectly. No notes.",
    "A bit tricky to set up, but the docs were great.",
    "My favorite feature by far. So useful.",
    "I would 100% recommend this.",
    "Solved my problem in just a few minutes."
]

NEGATIVE_COMMENTS = [
    "I'm so frustrated. This is completely broken.",
    "The instructions are confusing and I'm stuck.",
    "This is the worst product I have ever used. Terrible.",
    "Doesn't work. The button does nothing.",
    "I've been trying for an hour and it's impossible.",
    "The old version was so much better. This is awful."
]

NUM_ENTRIES = 100

print(f"Starting to send {NUM_ENTRIES} feedback entries for the last 30 days to the API...")

now = datetime.now()

for i in range(NUM_ENTRIES):
    resource = random.choice(RESOURCES)

    # Make some entries positive, some negative
    if random.random() > 0.4: #60% chance positive
        rating = random.randint(7, 10)
        comment = random.choice(POSITIVE_COMMENTS)
    else: #40% chance negative
        rating = random.randint(1, 4)
        comment = random.choice(NEGATIVE_COMMENTS)


    # Generate a random number of days ago within the last 30 days
    days_ago = random.randint(0, 30)

    # Subtract that from the current time
    fake_date = now - timedelta(days=days_ago)

    # Convert the date to ISO format string
    fake_date_str = fake_date.isoformat()

    # Create the feedback entry (data payload)
    feedback_entry = {
        "resource_id": resource,
        "rating": rating,
        "comment": comment,
        "created_at": fake_date_str
    }

    # Send the feedback to the API (POST request)
    try:
        response = requests.post(API_URL, json=feedback_entry)
        response.raise_for_status()  # Raise an error for bad responses
        print(f"[{i+1}/{NUM_ENTRIES}] Sent feedback for '{resource}' with rating {rating}. Posted: {fake_date}")
    except requests.exceptions.RequestException as e:
        print(f"\nERROR: Could not post data. Is your API running?")
        print(e)
        break # Stop the script if the API is not reachable


    # Wait a bit before sending the next entry
    time.sleep(0.05) #50 milliseconds

    print("\nFinished sending feedback entries.")