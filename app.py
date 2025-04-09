import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained model and tokenizer
model_path = "./models/BERT_Multi-Label_classification"  # Path to your saved model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline for multi-label classification
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # Use CPU
    return_all_scores=True  # Return scores for all labels
)

# Streamlit app title
st.title("Multi-Label Text Classification")

# Input text box
user_input = st.text_area("Enter text for classification:", height=150)

# Button to trigger inference
if st.button("Classify Text"):
    if user_input.strip():
        # Start timing
        start_time = time.time()

        # Perform inference
        results = classifier(user_input, truncation=True, padding="max_length", max_length=128)

        # End timing
        end_time = time.time()
        inference_time = end_time - start_time

        # Process results
        labels = [result["label"] for result in results[0]]
        scores = [result["score"] for result in results[0]]

        # Display results as a bar plot
        st.subheader("Classification Results")
        fig, ax = plt.subplots()
        ax.barh(labels, scores, color="skyblue")
        ax.set_xlabel("Confidence Score")
        ax.set_title("Label Confidence Scores")
        st.pyplot(fig)

        # Display inference time
        st.subheader("Inference Time")
        st.write(f"Inference completed in {inference_time:.4f} seconds.")
    else:
        st.warning("Please enter some text for classification.")