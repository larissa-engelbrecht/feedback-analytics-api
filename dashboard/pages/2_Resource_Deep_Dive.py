import streamlit as st
import requests
import pandas as pd

# Define API URLs
STATS_URL = "http://localhost:8002/feedback/stats/"
FEEDBACK_URL = "http://localhost:8002/feedback/"
SUMMARIZE_URL = "http://localhost:8002/feedback/summarize/"

st.markdown("""
<style>

/* Make the WHOLE selectbox clickable with pointer cursor */
div[data-baseweb="select"] {
    cursor: pointer !important;
}

div[data-baseweb="select"] > div {
    cursor: pointer !important;
}            

/* Make the internal button area a pointer too */
div[data-baseweb="select"] div[role="button"] {
    cursor: pointer !important;
}

/* Make arrow icon also pointer */
div[data-baseweb="select"] svg {
    cursor: pointer !important;
}

</style>
""", unsafe_allow_html=True)



# --- Page Configuration ---
st.set_page_config(
    page_title="Resource Deep Dive",
    page_icon=":material/search:",
    layout="wide",
    
)

st.title("Resource Deep Dive")
st.markdown("Analyze the feedback for a single resource and get an AI-powered summary.")

# --- Data Loading ---
@st.cache_data
def load_stats_data():
    try:
        response = requests.get(STATS_URL)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error laoding stats data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_feedback_data():
    try:
        response = requests.get(FEEDBACK_URL)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        if not df.empty:
            df['created_at'] = pd.to_datetime(df["created_at"])
        return df
    except Exception as e:
        st.error(f"Error laoding stats data: {e}")
        return pd.DataFrame()
    
@st.cache_data
def get_ai_summary(resource_id):
    """Calls the AI endpoint and caches the result."""
    try:
        payload = {"resource_id": resource_id}
        response = requests.post(SUMMARIZE_URL, json=payload)
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            st.error(f"AI Error: {data['error']}")
            return None
        
        return data.get("summary")
    except Exception as e:
        st.error(f"Failed to get AI summary: {e}")
        return None

# --- Main Page Logic ---
df_stats = load_stats_data()
df_feedback = load_feedback_data()

if df_stats.empty or df_feedback.empty:
    st.warning("No data available.")
else:
    # Select Resource
    resource_list = df_stats['resource_id'].unique()

    selected_resource = st.selectbox(
        "Select a Resource to Analyze:",
        options = resource_list,
        index = None,
        placeholder="Choose a resource...",
    )

    if selected_resource:
        
        st.subheader(f"AI Summary for {selected_resource}")
        
        with st.spinner("Calling AI model... This may take a few seconds."):
            ai_summary = get_ai_summary(selected_resource)
            if ai_summary:
                st.info(ai_summary)
            else:
                st.warning("No summary available for this resource.")

        st.divider()

        # Show the table header
        st.subheader("All Feedback Entries for this Resource")

        df_filtered = df_feedback[df_feedback['resource_id'] == selected_resource].sort_values(
            by="created_at", ascending=False
        )

        st.dataframe(
            df_filtered,
            width='stretch',
            hide_index=True,
            column_config={
                "id": None,
                "resource_id": st.column_config.TextColumn(label="Resource"),
                "created_at": st.column_config.DatetimeColumn(
                    label="Date Submitted",
                    format="YYYY-MM-DD HH:mm"
                ),
                "rating": st.column_config.NumberColumn(label="Rating"),
                "comment": st.column_config.TextColumn(label="Comment"),
                "sentiment": st.column_config.TextColumn(label="Sentiment"),
            }
        )