# ==========================================
# AI CUSTOMER CHAT ANALYSIS STREAMLIT APP
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from transformers import pipeline
from collections import Counter

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Customer Chat Analysis",
    layout="wide"
)

st.title("🤖 AI Customer Chat Analysis Dashboard")

# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    df = pd.read_csv("customer_chats_200.csv")

    return df

df = load_data()

# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text

df["clean_message"] = df["message"].apply(clean_text)

# ==========================================
# LOAD HUGGINGFACE MODEL
# ==========================================

@st.cache_resource
def load_model():

    classifier = pipeline("sentiment-analysis")

    return classifier

classifier = load_model()

# ==========================================
# SENTIMENT FUNCTION
# ==========================================

def hf_sentiment(text):

    result = classifier(text)[0]

    label = result["label"]

    if label == "POSITIVE":
        return "Positive"

    elif label == "NEGATIVE":
        return "Negative"

    else:
        return "Neutral"

# ==========================================
# APPLY SENTIMENT
# ==========================================

with st.spinner("Running AI Sentiment Analysis..."):

    df["sentiment"] = df["clean_message"].apply(
        hf_sentiment
    )

# ==========================================
# TOPIC CLASSIFICATION
# ==========================================

TOPIC_KEYWORDS = {

    "Delivery Issue": [
        "deliver", "ship", "arrival", "delay",
        "late", "tracking"
    ],

    "Refund / Payment": [
        "refund", "charge", "payment",
        "money", "billing"
    ],

    "Product Quality": [
        "broken", "quality", "defect",
        "damage", "poor"
    ],

    "Account / Login": [
        "login", "account",
        "password", "reset"
    ],

    "App / Website": [
        "app", "website",
        "crash", "bug"
    ],

    "Order Management": [
        "order", "cancel",
        "status", "return"
    ],

    "Customer Service": [
        "agent", "support",
        "service", "staff"
    ],

    "General Feedback": [
        "love", "great",
        "excellent", "amazing",
        "happy"
    ]
}

def categorize_topic(text):

    text = text.lower()

    for topic, keywords in TOPIC_KEYWORDS.items():

        for keyword in keywords:

            if keyword in text:

                return topic

    return "Other"

df["topic"] = df["clean_message"].apply(
    categorize_topic
)

# ==========================================
# COUNTS
# ==========================================

sent_counts = df["sentiment"].value_counts()

topic_counts = df["topic"].value_counts()

# ==========================================
# DASHBOARD
# ==========================================

col1, col2 = st.columns(2)

# ==========================================
# PIE CHART
# ==========================================

with col1:

    st.subheader("Sentiment Distribution")

    fig1, ax1 = plt.subplots()

    ax1.pie(
        sent_counts.values,
        labels=sent_counts.index,
        autopct="%1.1f%%"
    )

    st.pyplot(fig1)

# ==========================================
# TOPIC BAR CHART
# ==========================================

with col2:

    st.subheader("Topic Distribution")

    fig2, ax2 = plt.subplots()

    sns.barplot(
        x=topic_counts.values,
        y=topic_counts.index,
        ax=ax2
    )

    st.pyplot(fig2)

# ==========================================
# SENTIMENT PER TOPIC
# ==========================================

st.subheader("Sentiment per Topic") 
pivot = df.groupby( ["topic", "sentiment"] ).size().unstack(fill_value=0)

fig3, ax3 = plt.subplots( figsize=(10,5) )

pivot.plot( kind="bar", ax=ax3 ) 

st.pyplot(fig3)

# ==========================================
# DATASET VIEW
# ==========================================

st.subheader("Processed Dataset")

st.dataframe(df)

# ==========================================
# REAL-TIME PREDICTION
# ==========================================

st.subheader("Real-Time Customer Message Analysis")

user_input = st.text_area(
    "Enter Customer Message"
)

if st.button("Analyze"):

    clean = clean_text(user_input)

    sentiment = hf_sentiment(clean)

    topic = categorize_topic(clean)

    st.success(f"Predicted Sentiment: {sentiment}")

    st.info(f"Predicted Topic: {topic}")

