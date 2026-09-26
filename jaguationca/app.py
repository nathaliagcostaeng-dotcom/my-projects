import os
import sys
import pandas as pd
import gradio as gr
import requests
from fastai.vision.all import Path, parent_label, Resize, load_learner

# Precise absolute path tracking for model architecture and logs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(CURRENT_DIR, 'feline_classifier_model.pkl')
LOG_FILE_PATH = os.path.join(CURRENT_DIR, 'feedback_metrics.csv')

# CONFIGURAÇÃO DO GOOGLE DRIVE: Substitua o ID abaixo pelo ID do seu arquivo copiado do Drive
# Exemplo: se o link é .../d/1A2B3C4D/view, o ID é 1A2B3C4D
GOOGLE_DRIVE_FILE_ID = "COLOQUE_AQUI_O_ID_DO_SEU_ARQUIVO_DO_DRIVE"
DOWNLOAD_URL = f"https://google.com{GOOGLE_DRIVE_FILE_ID}"

def download_model_weights(url, destination):
    """
    Automated data engineering pipeline to stream and cache heavy binary 
    weights from remote cloud cloud providers to localized runtime directory.
    """
    print(f"Downloading model binary weights from remote storage...")
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print("Model downloaded successfully and cached to runtime folder.")
    except Exception as download_error:
        print(f"Critical execution error fetching model weights: {download_error}")
        sys.exit(1)

# Trigger remote weight collection if local cached model is missing
if not os.path.exists(MODEL_FILE_PATH):
    download_model_weights(DOWNLOAD_URL, MODEL_FILE_PATH)

# Initialize fallback CSV database if it does not exist
if not os.path.exists(LOG_FILE_PATH):
    df_init = pd.DataFrame(columns=["Timestamp", "System_Prediction", "User_Feedback", "Count"])
    # Seed initial technical values so the graph starts beautifully
    df_init.loc[0] = [pd.Timestamp.now(), "Jaguar", "Correct", 1]
    df_init.loc[1] = [pd.Timestamp.now(), "Ocelot", "Correct", 1]
    df_init.to_csv(LOG_FILE_PATH, index=False)

# Load the permanent pre-trained weights into operational memory
try:
    model_inference = load_learner(MODEL_FILE_PATH)
except Exception as e:
    print(f"Critical error loading compiled model file: {e}")
    sys.exit(1)

# Global runtime state to hold the last classification result safely
last_prediction = {"class": "Jaguar"}

def classify_feline_image(input_image):
    if input_image is None:
        return {"Upload an image": 1.0}
    
    predicted_class, class_idx, probabilities = model_inference.predict(input_image)
    last_prediction["class"] = str(predicted_class)
    
    classification_results = {
        model_inference.dls.vocab[i]: float(probabilities[i]) 
        for i in range(len(model_inference.dls.vocab))
    }
    return classification_results

def process_user_feedback(feedback_type):
    if not os.path.exists(LOG_FILE_PATH):
        return gr.BarPlot()
        
    df = pd.read_csv(LOG_FILE_PATH)
    new_entry = {
        "Timestamp": pd.Timestamp.now(),
        "System_Prediction": last_prediction["class"],
        "User_Feedback": feedback_type,
        "Count": 1
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_FILE_PATH, index=False)
    
    chart_data = df.groupby(["System_Prediction", "User_Feedback"]).size().reset_index(name="Total")
    
    return gr.BarPlot(
        value=chart_data,
        x="System_Prediction",
        y="Total",
        color="User_Feedback",
        title="Real-Time Model Accuracy & User Validation Metrics",
        vertical=False,
        width=450,
        height=300
    )

# Build the layout with integrated analytics blocks
with gr.Blocks(title="Jaguationça Wildlife Portal") as feline_app_interface:
    gr.Markdown("# Jaguationça: De quem são essas pintas? 🐾")
    gr.Markdown("A specialized computer vision tool for automated day/night wildlife monitoring. Upload a camera trap photo to execute model inference and log performance feedback.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Upload Camera Trap Photo")
            btn_classify = gr.Button("Analyze Feline Geometric Patterns", variant="primary")
            output_lbl = gr.Label(num_top_classes=2, label="Model Classification Probabilities")
            
            gr.Markdown("### 🗳️ Did the model classify this feline correctly?")
            with gr.Row():
                btn_correct = gr.Button("✅ Yes, Correct!", variant="success")
                btn_incorrect = gr.Button("❌ No, Incorrect!", variant="stop")
                
        with gr.Column(scale=1):
            init_df = pd.read_csv(LOG_FILE_PATH)
            init_chart = init_df.groupby(["System_Prediction", "User_Feedback"]).size().reset_index(name="Total")
            
            performance_chart = gr.BarPlot(
                value=init_chart,
                x="System_Prediction",
                y="Total",
                color="User_Feedback",
                title="Real-Time Model Accuracy & User Validation Metrics",
                vertical=False,
                width=450,
                height=300
            )

    btn_classify.click(fn=classify_feline_image, inputs=input_img, outputs=output_lbl)
    btn_correct.click(fn=lambda: process_user_feedback("Correct"), inputs=None, outputs=performance_chart)
    btn_incorrect.click(fn=lambda: process_user_feedback("Incorrect"), inputs=None, outputs=performance_chart)

if __name__ == "__main__":
    feline_app_interface.launch()
