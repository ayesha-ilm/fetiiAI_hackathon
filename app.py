# -*- coding: utf-8 -*-
"""
Streamlit + DuckDB + Mistral API for Fetii Austin Rideshare
"""

import streamlit as st
import pandas as pd
import duckdb
import numpy as np
from mistralai import Mistral
import datetime

# --- Current date ---
today = datetime.datetime.now().strftime("%Y-%m-%d")

# --- 1️⃣ Mistral API setup ---
api_key = st.secrets["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small"

# --- 2️⃣ Helper functions ---
def get_embeddings(text_list):
    response = client.embeddings.create(model=EMBED_MODEL, inputs=text_list)
    return [item.embedding for item in response.data]

def query_llm(messages: list) -> str:
    response = client.chat.complete(model=LLM_MODEL, messages=messages)
    return response.choices[0].message.content

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- 3️⃣ Load DuckDB ---
@st.cache_resource
def get_connection():
    # keep connection alive across reruns
    return duckdb.connect("fetii.duckdb")

@st.cache_data
def load_venues(con):
    return con.execute("SELECT * FROM venues").df()

con = get_connection()
venues = load_venues(con)

# con = duckdb.connect("fetii.duckdb")
# venues = con.execute("SELECT * FROM venues").df()

# Compute min/max dates for context
trip_dates = con.execute("SELECT MIN(\"Trip Date and Time\"), MAX(\"Trip Date and Time\") FROM trips").fetchall()[0]
min_date, max_date = trip_dates

# Precompute embeddings (cached)
if "venue_embeddings" not in st.session_state:
    batch_size = 50
    embeddings = []
    # ensure addresses are strings
    all_texts = [str(addr) for addr in venues["all_addresses"]]
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        embeddings.extend(get_embeddings(batch))
    st.session_state["venue_embeddings"] = embeddings
venues["embedding"] = st.session_state["venue_embeddings"]

# --- 4️⃣ Streamlit UI ---
st.set_page_config(page_title="FetiiAI Chatbot", page_icon="🚐", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": f"You are a helpful rideshare assistant for FetiiAI that shares information about popular ride-sharing destinations, demographics, and trends. Today's date is {today}."}
    ]

# Sidebar controls
with st.sidebar:
    st.title("⚙️ Controls")
    if st.button("Clear Chat History"):
        st.session_state["messages"] = [
            {"role": "system", "content": f"You are a helpful rideshare assistant for FetiiAI that shares information about popular ride-sharing destinations, demographics, and trends. Today's date is {today}."}
        ]
        st.session_state["visualizations"] = []

# Initialize visualizations storage
if "visualizations" not in st.session_state:
    st.session_state["visualizations"] = []

st.title("🚐 Fetii Ridesharing Chatbot")

# --- Chat display ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Render visualization if attached
                if "plot_data" in msg:
                    st.bar_chart(msg["plot_data"])

# --- User input ---
if prompt := st.chat_input("Ask about trips, riders, or destinations..."):

    # 1️⃣ Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 2️⃣ Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3️⃣ Create assistant placeholder
    placeholder_index = len(st.session_state["messages"])
    st.session_state["messages"].append({"role": "assistant", "content": "Generating response..."})
    placeholder = st.empty()
    placeholder.markdown("Generating response...")

    # --- Trivial greetings ---
    trivial_inputs = ["hi", "hello", "hey", "thanks", "thank you"]
    if prompt.strip().lower() in trivial_inputs:
        response = "Hello! 👋 How can I help you with trips, riders, or destinations today?"
        st.session_state["messages"][placeholder_index]["content"] = response
        placeholder.markdown(response)

    else:
        # --- Intent classification (improved) ---
        intent_prompt = f"""
You are classifying user questions for a rideshare chatbot.

Rules:
- If the question contains the name of a venue, location, destination, 
  or mentions number of trips/groups, or contains a time reference 
  (like "last month", "yesterday", "this week", "in June"), 
  then it MUST be classified as DATA_QUERY.
- Only small-talk like "hi", "hello", or questions about the chatbot itself 
  are GENERAL.

Question: "{prompt}"

Return only DATA_QUERY or GENERAL.
"""



        intent = query_llm([{"role": "user", "content": intent_prompt}]).strip()
        print(intent)
        if "DATA_QUERY" in intent or "DATA\_QUERY" in intent:
            print("here")
            query_embedding = get_embeddings([prompt])[0]
            venues["similarity"] = venues["embedding"].apply(lambda x: cos_sim(x, query_embedding))
            top_matches = venues.sort_values("similarity", ascending=False).head(5)
            h3_list = top_matches["h3"].tolist()

            # --- Handle "last month" gracefully ---
            date_filter = ""
            user_timeframe_note = ""
            if "last month" in prompt.lower():
                # Compute last full calendar month
                last_month_start = (datetime.date.today().replace(day=1) - pd.DateOffset(months=1)).date()
                last_month_end = (datetime.date.today().replace(day=1) - pd.DateOffset(days=1)).date()

                # Clamp to dataset range
                query_start = max(pd.to_datetime(min_date).date(), last_month_start)
                query_end = min(pd.to_datetime(max_date).date(), last_month_end)

                if query_start > query_end:
                    # No overlap at all
                    user_timeframe_note = "The dataset does not contain any trips from last month."
                    date_filter = "AND 1=0"  # force empty
                else:
                    if query_start == query_end:
                        user_timeframe_note = f"Results only reflect data from {query_start:%b %d}, since the dataset does not fully cover last month."
                    elif (query_start != last_month_start) or (query_end != last_month_end):
                        user_timeframe_note = f"Results only reflect {query_start:%b %d} – {query_end:%b %d}, since the dataset does not fully cover last month."
                    else:
                        user_timeframe_note = f"Results reflect the full month of {last_month_start:%B %Y}."

                    date_filter = f"""
                    AND CAST("Trip Date and Time" AS DATE) BETWEEN '{query_start}' AND '{query_end}'
                    """

            # --- SQL Query ---
            sql = f"""
            SELECT *
            FROM trips
            WHERE (pickup_h3 IN ({', '.join([f"'{h}'" for h in h3_list])})
                OR dropoff_h3 IN ({', '.join([f"'{h}'" for h in h3_list])}))
                {date_filter}
            LIMIT 200
            """

            try:
                sample_trips = con.execute(sql).df()
            except Exception as e:
                response = f"Sorry, I was unable to execute the SQL query: {str(e)}"
                st.session_state["messages"][placeholder_index]["content"] = response
                placeholder.markdown(response)
                sample_trips = pd.DataFrame()

            # --- LLM summarization ---
            context = f"""
            You are FetiiAI, a rideshare assistant.
            Today's date is {today}.
            User query: "{prompt}"

            Found {len(sample_trips)} trips matching the query.
            {user_timeframe_note}

            Top matched venues: {top_matches['all_addresses'].tolist()}
            Sample trips: {sample_trips.to_dict(orient='records')}

            Instructions:
            - ONLY use the data provided above.
            - If no trips were found at all, say exactly: "No matching trips were found for this query."
            - If trips exist, summarize clearly: total groups, typical group sizes, and most common hours.
            - Mention timeframe note when relevant.
            - Do not expose dataset min/max dates directly.
            - Do not invent external info or trends.
            """

            llm_messages = [
                {"role": "system", "content": "You are FetiiAI, a rideshare assistant. Follow the instructions carefully."},
                {"role": "user", "content": context}
            ]
            response = query_llm(llm_messages)

            st.session_state["messages"][placeholder_index]["content"] = response
            placeholder.markdown(response)

            if not sample_trips.empty:
                sample_trips["hour"] = pd.to_datetime(sample_trips["Trip Date and Time"]).dt.hour
                hourly_counts = sample_trips.groupby("hour").size()
                st.bar_chart(hourly_counts)


        else:
            # --- GENERAL question ---
            response = query_llm(st.session_state["messages"] + [{"role": "user", "content": prompt}])
            st.session_state["messages"][placeholder_index]["content"] = response
            placeholder.markdown(response)
