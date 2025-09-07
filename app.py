import gradio as gr
import pandas as pd
import joblib

# Load Model & Data
model = joblib.load("model.pkl")
group_mapping = joblib.load("group_mapping.pkl")
symptom_list = joblib.load("symptom_list.pkl")

def predict_disease(symptoms):
    input_data = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    group = model.predict([input_data])[0]
    return group_mapping[group]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Disease Prediction from Symptoms")
    
    symptom_input = gr.CheckboxGroup(
        choices=symptom_list,
        label="Select your symptoms"
    )
    
    output = gr.Textbox(label="Predicted Disease")
    
    submit_btn = gr.Button("Predict")
    submit_btn.click(fn=predict_disease, inputs=symptom_input, outputs=output)

demo.launch()
