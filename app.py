import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from shap_util import vanilla_explainer, simCSE_explainer, label_map, vanilla_classifier, simCSE_binary_explainer
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
            # Start timing for simCSE explainer
            start_time_simCSE = time.time()
            simCSE_shap_values = simCSE_explainer([user_input])
            end_time_simCSE = time.time()
            inference_time_simCSE = end_time_simCSE - start_time_simCSE

            # Start timing for vanilla explainer
            start_time_vanilla = time.time()
            vanilla_shap_values = vanilla_explainer([user_input])
            end_time_vanilla = time.time()
            inference_time_vanilla = end_time_vanilla - start_time_vanilla

            # Start timing for simCSE binary explainer
            start_time_simBinary = time.time()
            simBinary_shap_values = simCSE_binary_explainer([user_input])
            end_time_simBinary = time.time()
            inference_time_simBinary = end_time_simBinary - start_time_simBinary

            # Display SHAP text plot for Binary SHAP Explanation
            st.subheader("Binary SHAP Explanation")
            simBinary_shap_html = f"<head>{shap.plots.text(simBinary_shap_values[0], display=False)}</head>"
            components.html(
                f"""
                <div style="background-color: white; padding: 10px;">
                    {simBinary_shap_html}
                </div>
                """,
                height=200,
                scrolling=True,
            )
            st.write(f"Inference Time: {inference_time_simBinary:.4f} seconds")

            # Display SHAP text plot for simCSE SHAP Explanation
            st.subheader("simCSE SHAP Text Explanation")
            simCSE_shap_html = f"<head>{shap.plots.text(simCSE_shap_values[0], display=False)}</head>"
            components.html(
                f"""
                <div style="background-color: white; color: black; padding: 10px;">
                    {simCSE_shap_html}
                </div>
                """,
                height=200,
                scrolling=True,
            )
            st.write(f"Inference Time: {inference_time_simCSE:.4f} seconds")

            # Display SHAP text plot for Vanilla SHAP Explanation
            st.subheader("Vanilla SHAP Text Explanation")
            vanilla_shap_html = f"<head>{shap.plots.text(vanilla_shap_values[0], display=False)}</head>"
            components.html(
                f"""
                <div style="background-color: white; color: black; padding: 10px;">
                    {vanilla_shap_html}
                </div>
                """,
                height=200,
                scrolling=True,
            )
            st.write(f"Inference Time: {inference_time_vanilla:.4f} seconds")
    else:
        st.warning("Please enter some text for classification.")