# FeedBackLoop: An End-to-End MLOps Feedback Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)](https://fastapi.tiangolo.com/) [![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red?logo=streamlit)](https://streamlit.io/) [![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://www.postgresql.org/)

A complete, AI-powered system for ingesting, analyzing, and improving user feedback. This project demonstrates a full-stack, end-to-end MLOps workflow, from data ingestion to model re-training.

**Live Demo:** [Link to your deployed Streamlit App]

![FeedbackFlow Dashboard GIF](httpsor-screenshot-link)

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Database:** PostgreSQL (hosted on Neon)
- **ML/AI:** Scikit-learn, Pandas, SQLModel, Google Generative AI

## Deployment

This project is deployed as two separate services:

- **Backend API (FastAPI):**

  - **Service:** [**Render**](https://render.com/)
  - **Status:** `Live`
  - **URL:** `https://feedback-analytics-api.onrender.com`

- **Frontend Dashboard (Streamlit):**
  - **Service:** [**Streamlit Community Cloud**](https://share.streamlit.io/)
  - **Status:** `Live`
  - **URL:** `https://feedbackloop-api.streamlit.app/`

---

## The Problem

In many companies, user feedback from products, documents, or apps is unstructured and unmanaged. It's difficult to analyze at scale, identify trends, or route critical issues (like bug reports) to the right teams.

## The Solution

This platform solves the problem by providing a complete, self-improving system:

1.  **Ingest:** A high-speed **FastAPI** backend captures feedback from any source.
2.  **Analyze:** A custom-trained **scikit-learn** model automatically predicts sentiment (`positive`, `negative`).
3.  **Visualize:** An interactive **Streamlit** dashboard displays trends, KPIs, and AI-powered summaries (using the Gemini API).
4.  **Improve:** A "human-in-the-loop" **MLOps Validator** page allows humans to correct model mistakes.
5.  **Re-Train:** A training script downloads this "gold-standard" human-verified data to train and deploy new, smarter models.
6.  **Alert:** A **Webhook** system automatically sends alerts for negative feedback to external services (like Slack or Jira).

---

## Key Features

- **REST API (FastAPI):** Asynchronous API for high-performance ingestion of new feedback.
- **Custom Sentiment Model (scikit-learn):** A custom `LogisticRegression` model trained on product review data, replacing generic libraries for higher accuracy.
- **AI-Powered Summaries (Google Gemini):** A "Deep Dive" page that uses the Gemini API to provide on-demand AI summaries for all feedback on a specific product.
- **Interactive Analytics Dashboard (Streamlit):**
  - Tracks KPIs (Total Feedback, Avg. Rating).
  - Shows "Best & Worst" product leaderboards.
  - Visualizes feedback trends over time.
- **Full MLOps Re-Training Loop:**
  - **Validate:** A UI to correct the model's predictions.
  - **Re-Train:** A `train.py` script that queries the database for verified data.
  - **Deploy:** A simple process to promote the `v2` model to the live API.
  - **Webhook Alert System:** Allows users to subscribe to events (e.g., "negative_only") and receive alerts at a target URL.

---

## A Note on the Deployed Model

This project successfully demonstrates a full MLOps re-training pipeline. The "MLOps Validator" page correctly saves human-verified data, and the `train.py` script can use this data to build a new `v2` model.

However, for the live demo, the API is intentionally using the **`v1` model** (trained on 50,000+ IMDB records).

The `v2` model (trained on the small, human-verified dataset) is not yet in production because its training data is too small to be more accurate than the generalized `v1` model. As more data is validated, the `v2` model will improve and can be promoted to production by simply updating the `MODEL_PATH` in the API.

---

## How to Run Locally

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/feedback-analytics-api.git](https://github.com/your-username/feedback-analytics-api.git)
    cd feedback-analytics-api
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate   # (or source venv/bin/activate on Linux / macOS)
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**

    - Create a file named `.env` in the root folder.
    - Add your database URL and API key:

    ```
    DATABASE_URL="postgresql://user:pass@host/db"
    GEMINI_API_KEY="your-gemini-key"
    ```

4.  **Run the API:**

    ```bash
    uvicorn api.main:app --reload --port 8002
    ```

5.  **Run the Streamlit Dashboard (in a second terminal):**
    ```bash
    streamlit run dashboard/1_Home.py
    ```
