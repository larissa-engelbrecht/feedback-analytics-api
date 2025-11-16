import streamlit as st
import requests
import pandas as pd
import time

# --- HELPER FUNCTION ---
def get_sentiment_color(sentiment):
    """Returns a hex color code for a sentiment."""
    sentiment = str(sentiment).capitalize() # Ensure it's capitalized
    if sentiment == "Positive":
        return "#28a745" 
    elif sentiment == "Negative":
        return "#dc3545" 
    elif sentiment == "Neutral":
        return "#ffc107"
    else:
        return "#ffffff" 

# --- Define API URLs ---
FEEDBACK_URL = "http://localhost:8002/feedback/"
VALIDATE_URL = "http://localhost:8002/feedback/{feedback_id}/validate/"

# --- Page Configuration ---
st.set_page_config(
    page_title = "MLOps Validator",
    page_icon= ":material/smart_toy:",
    layout="wide"
)

# --- Data Loading Function (same as Home page) ---
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
        st.error(f"Error loading stats data: {e}")
        return pd.DataFrame()
    
# --- API Call Function ---
def validate_sentiment(feedback_id, new_sentiment):
    """Calls our new PUT endpoint to validate a sentiment."""
    try:
        url = VALIDATE_URL.format(feedback_id=feedback_id)
        payload = {"verified_sentiment": new_sentiment}
        response = requests.put(url, json=payload)
        response.raise_for_status()
        
        st.cache_data.clear() # Clear all of Streamlit's old data
        st.rerun() #  Force the page to reload from the top
    except Exception as e:
        st.error(f"Failed to validate sentiment: {e}")
        return False
    
# --- Header ---
st.title(":material/smart_toy: MLOps Validator")
st.markdown("""This page allows you to correct the model's predictions to create a high-quality dataset for re-training.""")

if st.button(":material/autorenew: Refresh Data"):
    st.cache_data.clear()
    st.session_state.current_index = 0 # Reset index on refresh
    st.rerun()

# --- Main Page Logic --- 
df = load_feedback_data()

if df.empty:
    st.warning("No feedback data available to validate.")
else:
    
    df_verified = df[df['verified_sentiment'].notnull()]
    df_unverified = df[df['verified_sentiment'].isnull()]


    # --- 2. Display the Stats ---
    st.subheader("Validation Progress")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Entries", len(df))
    col2.metric("Entries Validated", len(df_verified))
    col3.metric("Pending Validation", len(df_unverified))
    
    st.progress(len(df_verified) / len(df))
    st.divider()

    show_verified = st.checkbox("Show already verified entries")

    if show_verified:
        df_to_show = df_verified.sort_values(by="created_at", ascending=False)
        st.subheader(f"Editing: Already Validated ({len(df_to_show)})")
    else:
        df_to_show = df_unverified
        st.subheader(f"Pending Validation ({len(df_to_show)})")
    
   # --- 3. Build the "List View" of Cards ---
    #st.subheader(f"Items Pending Validation ({len(df_unverified)})")

    if df_to_show.empty:
        if show_verified:
            st.info("No items have been verified yet.")
        else:
            st.success("All feedback has been validated! You are ready to re-train.")
    
    # Loop through each unverified row and create a "card"
    for index, item in df_to_show.iterrows():
        
        with st.container(border=True):
            
            # --- Row 1: Comment and Prediction (Side-by-Side) ---
            col1, col2 = st.columns([2, 1]) 

            with col1:
                # --- Left Side: The Comment ---
                st.markdown(f"**Comment:**")
                st.markdown(f"> {item['comment']}")
            
            with col2:
                # --- Right Side: The (Colored) Prediction ---
                st.markdown("**Model's Prediction:**")
                sentiment_str = str(item['sentiment']).capitalize()
                color = get_sentiment_color(sentiment_str)
                st.markdown(f"""
                <div style="font-size: 1.75rem; font-weight: 600; color: {color};">
                    {sentiment_str}
                </div>
                """, unsafe_allow_html=True)
                
                # --- THIS IS THE NEW PART ---
                # If we are in "show_verified" mode, also show the verified label
                if show_verified and pd.notnull(item['verified_sentiment']):
                    st.markdown("**Human-Verified As:**")
                    verified_str = str(item['verified_sentiment']).capitalize()
                    verified_color = get_sentiment_color(verified_str)
                    st.markdown(f"""
                    <div style="font-size: 1.25rem; font-weight: 600; color: {verified_color};">
                        {verified_str}
                    </div>
                    """, unsafe_allow_html=True)
                # --- END NEW PART ---

            # --- Row 2: Divider and Buttons (Full Width) ---
            st.divider()

            st.markdown("**Set/Correct Verification:**") # Renamed for clarity
            b_col1, b_col2, b_col3 = st.columns(3)

            key_pos = f"pos_{item['id']}"
            key_neg = f"neg_{item['id']}"
            key_neu = f"neu_{item['id']}"

            if b_col1.button(":material/thumb_up: Set as Positive", key=key_pos, use_container_width=True):
                validate_sentiment(item['id'], 'positive')

            if b_col2.button(":material/thumb_down: Set as Negative", key=key_neg, use_container_width=True):
                validate_sentiment(item['id'], 'negative')

            if b_col3.button(":material/thumbs_up_down: Set as Neutral", key=key_neu, use_container_width=True):
                validate_sentiment(item['id'], 'neutral')