# -*- coding: utf-8 -*-
"""
Streamlit + DuckDB + Mistral API for Fetii Austin Rideshare
"""

import streamlit as st
import pandas as pd
import duckdb
import os
from mistralai import Mistral

# --- 1️⃣ Mistral API setup ---
# api_key = os.environ["iedLLDsnr1h06viHpRah1xOrCZ1eVfOn"]
client = Mistral(api_key="iedLLDsnr1h06viHpRah1xOrCZ1eVfOn")

EMBED_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small"  # or another LLM endpoint Mistral offers
# print(dir(client.chat))

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

def get_embeddings(text_list):
    """Return embeddings for a list of texts using Mistral embeddings API."""
    response = client.embeddings.create(model=EMBED_MODEL, inputs=text_list)
    return [item.embedding for item in response.data]
def query_llm(prompt: str) -> str:
    """Generate text using Mistral chat API."""
    messages.append({"role": "user", "content": prompt})

    response = client.chat.complete(
        model=LLM_MODEL,
        messages=messages,
        # max_tokens=512  # if you want to control output length
    )
    reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    return reply

# --- 2️⃣ Load DuckDB connection ---
con = duckdb.connect('fetii.duckdb')

# --- 3️⃣ Load venues table ---
venues = con.execute("SELECT * FROM venues").df()

# Precompute embeddings if not already
batch_size = 50  # or whatever is allowed by the API
embeddings = []

all_texts = [" ".join(addr) for addr in venues['all_addresses']]

for i in range(0, len(all_texts), batch_size):
    batch = all_texts[i:i+batch_size]
    batch_embeddings = get_embeddings(batch)
    embeddings.extend(batch_embeddings)

venues['embedding'] = embeddings

# --- 4️⃣ Streamlit UI ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("Fetii Austin Rideshare Chatbot (LLM + DuckDB + Mistral)")

query = st.text_input("Ask a question about trips, riders, or destinations:")

if st.button("Run Query") and query:

    # --- 4a: Semantic search for venues ---
    query_embedding = get_embeddings([query])[0]
    
    # Compute cosine similarity manually since we're using raw vectors
    import numpy as np
    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    venues['similarity'] = venues['embedding'].apply(lambda x: cos_sim(x, query_embedding))
    top_matches = venues.sort_values('similarity', ascending=False).head(5)
    
    # Collect all H3 hexes
    h3_list = top_matches['h3'].tolist()

    # --- 4b: Query trips table for matching pickup OR dropoff ---
    sql = f"""
    SELECT *
    FROM trips
    WHERE pickup_h3 IN ({', '.join([f"'{h}'" for h in h3_list])})
       OR dropoff_h3 IN ({', '.join([f"'{h}'" for h in h3_list])})
    LIMIT 50
    """
    sample_trips = con.execute(sql).df()
    trip_count = len(sample_trips)

    # --- 4c: Prepare prompt for LLM ---
    prompt = f"""
User query: "{query}"
Found {trip_count} trips matching top venues: {top_matches['all_addresses'].tolist()}
Sample trips: {sample_trips.to_dict(orient='records')}
Please summarize these trips in a human-friendly explanation,
highlighting popular pickup/drop-off locations, hours, and group sizes.
"""

    # --- 4d: Generate LLM response ---
    response = query_llm(prompt)

    st.subheader("LLM Summary")
    st.write(response)

    # --- 4e: Optional visualization ---
    if not sample_trips.empty:
        st.subheader("Trips by Hour")
        sample_trips['hour'] = pd.to_datetime(sample_trips['Trip Date and Time']).dt.hour
        hourly_counts = sample_trips.groupby('hour').size()
        st.bar_chart(hourly_counts)
