
import streamlit as st
import requests
import pandas as pd

# URL of the backend API
API_URL = "http://localhost:8002/feedback/"
STATS_URL = "http://localhost:8002/feedback/stats/"

# --- Page Configuration ---
st.set_page_config(
    page_title="Feedback Dashboard",
    page_icon=":material/bar_chart:",
    layout="wide",
)
# --- Page Title ---
st.title("FeedbackLoop Dashboard")
st.markdown("This is the central dashboard for analyzing all user feedback.")

# --- Refresh Data ---
if st.button("Refresh Data"):
    st.cache_data.clear() # Clear cached data to force reload
    st.rerun()

# --- Fetch Feedback Data from API ---
# Cache the data to avoid repeated API calls, when we press a button
@st.cache_data
def load_feedback_data():

    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()

        # Convert the list of feedback entries(dictionaries) to a DataFrame (Pandas)
        df = pd.DataFrame(data)
        if not df.empty:
            # Convert created_At to ddatetime for better sorting/filtering
            df['created_at'] = pd.to_datetime(df["created_at"])
        return df
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the backend API at {API_URL}. Please ensure the API is running.")
        return pd.DataFrame()  # Return an empty DataFrame on error
    except Exception as e:
        st.error(f"An error occured while fetching data: {e}")
        return pd.DataFrame() 

@st.cache_data
def load_feedback_stats_data():
    try:
        response = requests.get(STATS_URL)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        return pd.DataFrame(data)  
    except Exception as e:
        st.error(f"An error occured while fetching stats data: {e}")
        return pd.DataFrame()

# Load all data
df_feedback = load_feedback_data()
df_stats = load_feedback_stats_data()

# --- MAIN DASHBOARD ---
if df_feedback.empty:
    st.warning("No feedback data to display. Post some feedabck to your API")
else:
    # --- KPI Metrics ---
    st.header("High-Level KPIs")

    # Create 3 columns
    kpi1, kpi2, kpi3 = st.columns(3)

    total_feedback = len(df_feedback)
    avg_rating = df_feedback['rating'].mean()
    negative_count = (df_feedback['sentiment'] == 'negative').sum()

    kpi1.metric(
        label = "Total Feedback Entries",
        value=  f"{total_feedback}"
    )
    kpi2.metric(
        label = "Overall Average Rating",
        value = f"{avg_rating:.2f} / 10"
    )
    kpi3.metric(
        label = "Total Negative Comments",
        value = f"{negative_count}"
    )

    st.divider() # Adds a horizontal line

    # --- Best & Worst Resources ---
    st.header("Resource Leaderboards")

    if df_stats.empty:
        st.warning("No statistics data to display.")
    else:
        # Create 2 columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 5 Resources")
            df_best = df_stats.sort_values(by="average_rating", ascending=False).head(5)
            st.dataframe(
                df_best, 
                width='stretch',
                hide_index=True, 
                column_config={
                    "resource_id": st.column_config.TextColumn(label="Resource"),
                    "average_rating": st.column_config.NumberColumn(
                        "Average Rating",
                        format="%0.2f",
                    ),
                    "feedback_count": st.column_config.NumberColumn(label="Feedback Count"),
                    "negative_comment_count": None,
                    "positive_comment_count": st.column_config.NumberColumn(label="Positive Comments"),
                }
            )

        with col2:
            st.subheader("Worst 5 Resources")
            df_worst = df_stats.sort_values(by='average_rating', ascending=True).head(5)
            st.dataframe(
                df_worst,  
                width='stretch',
                hide_index=True, 
                column_config={
                    "resource_id": st.column_config.TextColumn(label="Resource"),
                    "average_rating": st.column_config.NumberColumn(
                        "Average Rating",
                        format="%0.2f",
                    ),
                    "feedback_count": st.column_config.NumberColumn(label="Feedback Count"),
                    "negative_comment_count": st.column_config.NumberColumn(label="Negative Comments"),
                    "positive_comment_count": None,
                }
            )

    st.divider()

    # --- Detailed Feedback ---
    st.header("Feedback Over Time")

    # Resample data by day and count entries
    df_time = df_feedback.set_index('created_at').resample('D').count()['id'].rename("Feedback Count")
    st.line_chart(df_time)

    # Raw Feedback Data
    st.subheader("All Feedback Entries")
    st.dataframe(
        df_feedback,  
        width='stretch',
        hide_index=True, 
        column_config={
            "id": None,
            "resource_id": st.column_config.TextColumn(label="Resource"),
            "created_at": st.column_config.DatetimeColumn(
                label="Date Submitted",
                format="YYYY-MM-DD HH:mm" # Format the date nicely
            ),
            "rating": st.column_config.NumberColumn(label="Rating"),
            "comment": st.column_config.TextColumn(label="Comment"),
            "sentiment": st.column_config.TextColumn(label="Sentiment"),
        }
    )