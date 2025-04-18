import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from shap_util import classifier, explainer, label_map
import streamlit.components.v1 as components
import shap
import numpy as np

# Streamlit app title
st.title("Multi-Label Text Classification with SHAP Explanations")

# Input text box
user_input = st.text_area("Enter text for classification:", height=100)

# Button to trigger inference
if st.button("Classify Text"):
    if user_input.strip():
        with st.spinner("Classifying text... Please wait."):
            # Start timing
            start_time = time.time()

            # Perform inference
            results = classifier(user_input, truncation=True, padding="max_length", max_length=128)
            shap_values = explainer([user_input])
            # End timing
            end_time = time.time()
            inference_time = end_time - start_time

            # Process results
            labels = list(label_map.keys())
            scores = [result["score"] for result in results[0]]

            # Display results as a bar plot
            st.subheader("Classification Results")
            fig, ax = plt.subplots()
            ax.barh(labels, scores, color="skyblue")
            ax.set_xlim(0, 1.0)  # Set the range of the x-axis to be between 0 and 1.0
            ax.set_xlabel("Confidence Score")
            ax.set_title("Label Confidence Scores")
            st.pyplot(fig)

            # Display SHAP text plot
            st.subheader("SHAP Text Explanation")
            shap_html = f"<head>{shap.plots.text(shap_values[0], display=False)}</head>"

            # Render dynamic SHAP HTML in Streamlit with black text and white background
            components.html(
                f"""
                <div style="background-color: white; color: black; padding: 10px;">
                    {shap_html}
                </div>
                """,
                height=300,  # Adjust height as needed
                scrolling=True,
            )

            # Display inference time
            st.subheader("Inference Time")
            st.write(f"Inference completed in {inference_time:.4f} seconds.")
    else:
        st.warning("Please enter some text for classification.")