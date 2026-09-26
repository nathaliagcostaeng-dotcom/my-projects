import os
import pandas as pd
import gradio as gr
from fastai.vision.all import Path, parent_label, Resize, load_learner

# Precise absolute path tracking for model architecture and logs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE_PATH = os.path.join(CURRENT_DIR, 'feline_classifier_model.pkl')
LOG_FILE_PATH = os.path.join(CURRENT_DIR, 'feedback_metrics.csv')

# Initialize fallback CSV database if it does not exist
if not os.path.exists(LOG_FILE_PATH):
    df_init = pd.DataFrame(columns=["Timestamp", "System_Prediction", "User_Feedback", "Count"])
    # Seed initial technical values so the graph starts beautifully
    df_init.loc[0] = [pd.Timestamp.now(), "Jaguar", "Correct", 1]
    df_init.loc[1] = [pd.Timestamp.now(), "Ocelot", "Correct", 1]
    df_init.to_csv(LOG_FILE_PATH, index=False)

# Load the permanent pre-trained weights
try:
    model_inference = load_learner(MODEL_FILE_PATH)
except Exception as e:
    print(f"Error loading model pkl file: {e}")

# Global runtime state to hold the last classification result safely
last_prediction = {"class": "Jaguar"}

def classify_feline_image(input_image):
    if input_image is None:
        return {"Upload an image": 1.0}
    
    predicted_class, class_idx, probabilities = model_inference.predict(input_image)
    
    # Store globally to know what the user is giving feedback to
    last_prediction["class"] = str(predicted_class)
    
    classification_results = {
        model_inference.dls.vocab[i]: float(probabilities[i]) 
        for i in range(len(model_inference.dls.vocab))
    }
    return classification_results

def process_user_feedback(feedback_type):
    """
    Logs user verification tokens into the CSV database and computes 
    aggregated performance metrics for real-time visualization layers.
    """
    if not os.path.exists(LOG_FILE_PATH):
        return gr.BarPlot()
        
    # Append the new feedback data point
    df = pd.read_csv(LOG_FILE_PATH)
    new_entry = {
        "Timestamp": pd.Timestamp.now(),
        "System_Prediction": last_prediction["class"],
        "User_Feedback": feedback_type,
        "Count": 1
    }
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_csv(LOG_FILE_PATH, index=False)
    
    # Group and aggregate data to reconstruct the distribution bar chart
    chart_data = df.groupby(["System_Prediction", "User_Feedback"]).size().reset_index(name="Total")
    
    # Return updated interface components dynamically
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
            # Read initial dataset structure to pre-render the visual analytics framework
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

    # Establish interactive software event routing pipelines
    btn_classify.click(fn=classify_feline_image, inputs=input_img, outputs=output_lbl)
    
    # Map feedback triggers to update the charting engine data buffers dynamically
    btn_correct.click(fn=lambda: process_user_feedback("Correct"), inputs=None, outputs=performance_chart)
    btn_incorrect.click(fn=lambda: process_user_feedback("Incorrect"), inputs=None, outputs=performance_chart)

if __name__ == "__main__":
    feline_app_interface.launch()

