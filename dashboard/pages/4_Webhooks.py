import streamlit as st
import requests
import pandas as pd



# --- Load URLs from Secrets ---

# Check if we are in local mode
LOCAL_MODE = st.secrets.get("LOCAL_MODE", False) # Default to False if not found

# Set the single base URL
if LOCAL_MODE:
    API_BASE_URL = st.secrets["LOCAL_API_BASE_URL"]
else:
    API_BASE_URL = st.secrets["DEPLOYED_API_BASE_URL"]

#  Build all your endpoints from the base URL
WEBHOOKS_URL = f"{API_BASE_URL}/webhooks/"


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
    page_title="Webhook Management",
    page_icon=":material/webhook:",
    layout="wide",
)

# --- API Call Functions ---

@st.cache_data
def load_webhooks():
    """Fetches all current webhook subscriptions."""
    try:
        response = requests.get(WEBHOOKS_URL)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading webhooks: {e}")
        return pd.DataFrame()
    
def create_webhook(target_url, event_type, resource_id=None):
    """Calls the POST endpoint to create a new webhook."""
    try:
        payload = {
            "target_url": target_url,
            "event_type": event_type,
            "resource_id": resource_id if resource_id else None
        }
        response = requests.post(WEBHOOKS_URL, json=payload)
        response.raise_for_status()
        st.cache_data.clear() # Clear cache on success
        st.rerun() # Rerun to show the new webhook
    except Exception as e:
        st.error(f"Failed to create webhook: {e}")

def delete_webhook(webhook_id):
    """Calls the DELETE endpoint to remove a webhook."""
    try:
        url = f"{WEBHOOKS_URL}{webhook_id}" # Construct the specific URL
        response = requests.delete(url)
        response.raise_for_status()
        st.cache_data.clear() # Clear cache on success
        st.rerun() # Rerun to remove the webhook from the list
    except Exception as e:
        st.error(f"Failed to delete webhook: {e}")



# --- Header ---
st.title(":material/webhook: Webhook Management")
st.markdown("Create subscriptions to send alerts to external services (like Slack or Zapier) when new feedback arrives.")

# --- Form to Create a New Webhook ---
st.subheader("Create a New Webhook Subscription")

with st.form(key="create_webhook_form"):
    st.markdown("Select an event and provide a URL to send the alert to.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        event_type = st.selectbox(
            "Event Type",
            options=["negative_only", "all_feedback"],
            index=0
        )
    
    with col2:
        target_url = st.text_input(
            "Target URL",
            placeholder="https://hooks.slack.com/services/..."
        )
    
    # Optional field to filter by resource_id
    resource_id = st.text_input(
        "Resource ID (Optional)",
        placeholder="doc-101-install-guide",
        help="Only send alerts for this specific resource. Leave blank for all."
    )
    
    submit_button = st.form_submit_button("Create Subscription")

    if submit_button:
        if not target_url:
            st.warning("Please provide a Target URL.")
        else:
            create_webhook(target_url, event_type, resource_id)

st.divider()

# --- List of Existing Webhooks ---

st.subheader("Manage Existing Subscriptions")
df_webhooks = load_webhooks()

if df_webhooks.empty:
    st.info("No webhooks have been created yet.")
else:
    # Display each webhook in a container
    for index, row in df_webhooks.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([4, 1]) # 4/5 width for info, 1/5 for button
            
            with col1:
                st.markdown(f"**Event:** `{row['event_type']}`")
                st.markdown(f"**URL:** `{row['target_url']}`")
                if row['resource_id']:
                    st.markdown(f"**Resource:** `{row['resource_id']}`")
            
            with col2:
                # We need a unique key for every button
                button_key = f"delete_{row['id']}"
                if st.button("Delete", key=button_key, use_container_width=True, type="primary"):
                    delete_webhook(row['id'])
