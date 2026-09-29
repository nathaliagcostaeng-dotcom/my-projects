# 🐾 Project Jaguationça: Deep Learning for Automated Wildlife Monitoring

An end-to-end computer vision and Data-Centric MLOps pipeline engineered to automate species identification and resolve morphological data bottlenecks in wildlife tracking. The system is specialized in differentiating between two sympatric neotropical felines: the **Jaguar (*Panthera onca*)** and the **Ocelot (*Leopardus pardalis*)**.

Created in google colab

---

## 🔬 Biological Context & Computer Vision Challenge

In field conservation across neotropical biomes (such as the Pantanal, Cerrado, and Atlantic Forest), distinguishing sympatric felines from raw camera trap captures introduces a massive data-triaging bottleneck. Both species possess yellow coats with black spot patterns, making automated feature extraction highly susceptible to errors under night-vision, low-resolution, or overexposed infrared (IR) flash conditions.

This project optimizes a Deep Learning model to decode the distinct geometric markers of each species:
*   **Jaguar:** Large, open rosettes containing central black spots.
*   **Ocelot:** Elongated spots connecting into horizontal, chain-like bands or stripes.

### Morphological Reference Matrix

| Feature | Jaguar (*Panthera onca*) | Ocelot (*Leopardus pardalis*) |
| :--- | :--- | :--- |
| **Coat Pattern** | Large, open rosettes with central black spots. | Elongated spots merging into horizontal chain-like bands. |
| **Build & Scale** | Massive head, robust muscle structures, apex predator (56–96 kg). | Slender build, narrow facial structure, medium wildcat (8–15 kg). |
| **Tail Proportion** | Short tail relative to body size. | Elongated, sleek tail, highly prominent relative to body. |

---

## ⚙️ Core Architecture & Data-Centric Engineering

The pipeline implements a production-grade **Data-Centric AI** framework to ensure feature extraction stability across complex lighting bounds:

*   **Curated Data Ingestion:** The runtime repository is dynamically compiled by harvesting and filtering specialized wildlife assets from private **Google Photos** repositories, open-access Kaggle matrices, and the Fastai community ecosystem.
*   **Deep Learning Backbone:** Deployed a **ResNet50** Convolutional Neural Network leveraging *Transfer Learning* (ImageNet pre-trained weights). Its advanced structural *Bottleneck Layers* isolate high-frequency pattern variations within coat geometries.
*   **High-Resolution Tensors:** Input arrays are standard-scaled to **448x448 pixels** (`Resize(448)`). This doubles the spatial pixel density relative to default models, protecting boundary lines from infrared flash overexposure.
*   **Geometric Fidelity Constraints:** Real-time data augmentations strictly freeze artificial scale expansions (`max_zoom=1.0`) while permitting horizontal flips and slight rotations. This guarantees that the network learns the pure physical continuity of stripes without synthetic artifacts.
*   **Loss Smoothing Regularization:** Integrated **Label Smoothing Cross Entropy** (`LabelSmoothingCrossEntropy`). This penalizes mathematical overconfidence during backpropagation, smoothing optimization steps and reducing overfitting risks on noisy field backgrounds.

---

## 📊 Model Performance Metrics

The model achieved an outstanding global **Accuracy of 94.44%** over an unbiased validation set of 108 wildlife camera trap assets.

### Detailed Classification Report
```text
              precision    recall  f1-score   support

      jaguar       0.95      0.95      0.95        60
      ocelot       0.94      0.94      0.94        48

    accuracy                           0.94       108
   macro avg       0.94      0.94      0.94       108
weighted avg       0.94      0.94      0.94       108
```

### Confusion Matrix Insights
```text
Actual \ Predicted   jaguar   ocelot
  jaguar               57        3
  ocelot                3       45
```
*   **High Precision Consistency:** A balanced F1-score across both classes (0.95 for Jaguar, 0.94 for Ocelot) proves that class balancing strategies effectively neutralized statistical preference biases.
*   **Error Distribution:** Residual errors (3 instances per class) represent near-field physical occlusions or severe infrared light saturation where continuous geometric shapes are partially obscured.

---

## 🚀 Execution Guide & Deployment

This ecosystem is fully self-contained, reproducible, and ready for edge-testing via Google Colab.

### Local Environment Preparation
Ensure you have your raw imagery structured locally as follows before loading into runtime memory:
```text
📂 conservation_dataset/
├── 📂 jaguar/
│   └── *.jpg
└── 📂 ocelot/
    └── *.jpg
```

### Running the System
1. Open the provided notebook in Google Colab.
2. Drag and drop your local `conservation_dataset` (unzip the file) folder directly into the Colab side file-browser panel.
3. Run all cells sequentially. Section 1 will install `fastai` and `gradio` seamlessly.
4. Section 2 executes the 5-epoch training loop using discriminative weights matching the ResNet50 constraints.
5. Section 3 automatically compiles the weights into a localized binary (`feline_classifier_model.pkl`) and initializes an interactive web-app wrapper using a secure, public **Gradio Live URL** valid for field testing.

---

## 🤖 Engineering Acknowledgments & Autonomy
This notebook was developed using an AI Pair Programming methodology. While I operated as the lead Data Engineer and Project Manager—defining data curation requirements, formatting camera trap constraints, and performing diagnostics—the code structure, MLOps pipeline optimizations, and interface layout were co-authored in collaboration with an advanced AI assistant.

