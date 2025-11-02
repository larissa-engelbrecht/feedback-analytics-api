
import streamlit as st
import requests
import pandas as pd

# URL of the backend API
API_URL = "http://localhost:8002/feedback/"

# --- Page Configuration ---
st.set_page_config(
    page_title="Feedback Dashboard",
    page_icon="📊",
    layout="wide",
)

# --- Page Title ---
st.title("📊 FeedbackLoop Dashboard")
st.markdown("This is the central dashboard for analyzing all user feedback.")

# --- Fetch Feedback Data from API ---
# Cache the data to avoid repeated API calls, when we press a button
@st.cache_data
def load_data():

    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Convert the list of feedback entries(dictionaries) to a DataFrame (Pandas)
        df = pd.DataFrame(data)

        # Convert created_At to ddatetime for better sorting/filtering
        df['created_at'] = pd.to_datetime(df["created_at"])
        return df
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {API_URL}. Please ensure the API is running.")
        return pd.DataFrame()  # Return an empty DataFrame on error
    except Exception as e:
        st.error(f"An error occured while fetching data: {e}")
        return pd.DataFrame() 

# Load all data
df = load_data()

# --- Display Data ---
st.header("All Feedback Entries")
st.write("Here is the collected user feedback data, sorted by most recent entries.")

if not df.empty:
    # Sort by creation date, newest first
    st.dataframe(df.sort_values(by="created_at", ascending=False), use_container_width=True)
else:
    st.warning("No data available to display.")