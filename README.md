# GroundedCXR — Explainable AI for Chest X-Ray Classification

An AI-powered research prototype for classifying chest X-ray images into **Normal** and **Pneumonia** categories using deep learning, with visual explanations through Grad-CAM and a conversational AI assistant for understanding model outputs.

## Overview

GroundedCXR combines image classification, explainable AI, and an interactive web interface to make model predictions easier to inspect and understand.

The system uses a fine-tuned **ResNet18 convolutional neural network** to classify chest X-ray images. It also uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize image regions that contributed to a prediction.

A Streamlit interface allows users to upload an X-ray, view the predicted class and class probabilities, and inspect the corresponding Grad-CAM visualization. A conversational AI assistant is being integrated to explain predictions and answer educational questions about the model.

> **Important:** GroundedCXR is an educational and research prototype. It is not a medical diagnostic system and must not be used as a substitute for professional medical evaluation.

## Key Features

- **Chest X-ray classification:** Classifies uploaded images as `NORMAL` or `PNEUMONIA`.
- **Transfer learning:** Uses an ImageNet-pretrained ResNet18 model with selected layers fine-tuned for the classification task.
- **Prediction probabilities:** Displays the model's probabilities for both classes and the predicted class's confidence score.
- **Explainable AI:** Generates Grad-CAM heatmaps to help inspect which image regions influenced the model's prediction.
- **Interactive web application:** Provides an image-upload and prediction interface built with Streamlit.
- **Conversational AI assistant:** Designed to explain model outputs, confidence scores, Grad-CAM, and relevant technical concepts.
- **Modular Python codebase:** Separates model loading, inference, visualization, and application logic.

## System Architecture

```text
                Chest X-ray Image
                        |
                        v
              Image Preprocessing
                        |
                        v
                  ResNet18 Model
                        |
                        v
             Classification Output
               /              \
              v                v
       Predicted Class    Class Probabilities
              |
              v
           Grad-CAM
              |
              v
      Streamlit Web Interface
              |
              v
    Conversational AI Assistant
       (Explanation Layer)
```

The ResNet18 model performs the actual image classification. Grad-CAM provides a visualization of influential regions, while the conversational assistant explains the available output. The assistant does not replace or perform the image classification.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning framework | PyTorch |
| Model architecture | ResNet18 |
| Image processing | Pillow, Torchvision |
| Explainable AI | Grad-CAM |
| Web interface | Streamlit |
| Conversational AI | Google Gemini API integration in progress |
| Development environment | VS Code, Google Colab |
| Version control | Git and GitHub |

## Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney, available on Kaggle.

**Dataset:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset contains approximately 5,863 chest X-ray images categorized into:

- `NORMAL`
- `PNEUMONIA`

The images are organized into training, validation, and test subsets in the original dataset.

For this project, the model was trained using a training/validation split, with the separate test set retained for evaluation.

### Dataset preparation

The preprocessing pipeline includes:

1. Resizing images to 224 × 224 pixels.
2. Converting grayscale images to three channels.
3. Converting images into PyTorch tensors.
4. Normalizing image tensors using ImageNet mean and standard deviation values.

The dataset is not included in this repository because it contains thousands of images. Download it from the original source and arrange it locally if you need to reproduce the experiments.

## Model Details

### ResNet18

ResNet18 is a convolutional neural network that uses residual connections to support effective training of deep neural networks.

This project uses transfer learning:

1. Start with pretrained ResNet18 weights.
2. Replace the final fully connected layer with a two-class output layer.
3. Freeze most of the pretrained network.
4. Fine-tune selected layers and the classification head on the chest X-ray dataset.
5. Save the best checkpoint based on validation performance.

### Training configuration

| Parameter | Value |
|---|---|
| Architecture | ResNet18 |
| Pretraining | ImageNet |
| Input image size | 224 × 224 |
| Number of output classes | 2 |
| Training epochs | 5 |
| Training environment | Google Colab with GPU |
| Loss function | Class-weighted cross-entropy |
| Model checkpoint | `best_pneumonia_resnet18.pth` |

The trained checkpoint is stored locally and is not committed to GitHub.

## Model Evaluation

The model was evaluated on a separate test set containing 624 images.

| Metric | Result |
|---|---:|
| Test accuracy | 81.57% |
| Pneumonia precision | 77.67% |
| Pneumonia recall (sensitivity) | 98.97% |
| Pneumonia F1-score | 87.03% |
| Normal recall (specificity) | 52.56% |

### Confusion matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | Normal | Pneumonia |
|---|---:|---:|
| Normal | 123 | 111 |
| Pneumonia | 4 | 386 |

### Interpretation

The model identifies most pneumonia-positive images in the test set, but it also produces a substantial number of false-positive predictions for normal images.

The best validation accuracy reached approximately 99.23%, while test accuracy was 81.57%. This gap indicates that validation performance does not fully reflect generalization to the separate test set.

These results motivate further investigation into generalization, false positives, dataset characteristics, and robustness on external images.

**The reported results are experimental results on this dataset. They do not establish clinical accuracy or real-world medical reliability.**

## Explainability with Grad-CAM

GroundedCXR uses Gradient-weighted Class Activation Mapping (Grad-CAM) to generate heatmaps associated with model predictions.

Grad-CAM uses gradients flowing into a selected convolutional layer to highlight regions that contributed to a selected class score.

In this project, the target layer is:

```python
model.layer4[-1]
```

The resulting visualization is overlaid on the input image to make the model's behavior easier to inspect.

### Why Grad-CAM?

- Helps visualize image regions that influenced a prediction.
- Supports qualitative analysis of correct and incorrect predictions.
- Helps investigate potential model biases and unwanted image cues.
- Makes model behavior more interpretable for demonstrations and research.

**Limitation:** A Grad-CAM heatmap does not prove that pneumonia is present in the highlighted region. It is an interpretation of model behavior, not a clinical explanation.

## Project Structure

```text
GroundedCXR/
├── dataset/
│   └── chest_xray/              # Local dataset; excluded from Git
├── models/
│   └── best_pneumonia_resnet18.pth  # Local checkpoint; excluded from Git
├── src/
│   ├── __init__.py
│   ├── check_dataset.py
│   ├── data_loader.py
│   ├── model.py
│   ├── view_dataset.py
│   ├── inference.py
│   ├── gradcam.py
│   └── chatbot.py               # If present in the current version
├── app.py                       # Streamlit application
├── requirements.txt
├── .gitignore
└── README.md
```

The exact contents may vary depending on the current development stage. The dataset, virtual environment, API secrets, and trained checkpoint should remain outside version control.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/GroundedCXR.git
cd GroundedCXR
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly or use Command Prompt activation instead.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not yet present or does not include all required packages, install the packages used by the current application:

```bash
pip install torch torchvision streamlit pillow numpy "grad-cam>=1.5.0,<2.0" python-dotenv google-genai
```

Choose the PyTorch installation appropriate for your hardware from the official guide: https://pytorch.org/get-started/locally/

### 4. Add the trained model checkpoint

Place your trained checkpoint at:

```text
models/best_pneumonia_resnet18.pth
```

The checkpoint is not included in this GitHub repository. Use your locally saved checkpoint or obtain it separately from a trusted project backup.

### 5. Configure the conversational assistant

If the current version includes the Gemini chatbot, create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3-flash-preview
```

Obtain an API key through Google AI Studio:

https://aistudio.google.com/

Never commit `.env` or publish your API key. Gemini model access and free-tier limits may vary by account and model.

If the chatbot integration is not yet complete, the image classification and Grad-CAM functionality can be tested separately, provided the model checkpoint and required dependencies are available.

### 6. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Start the Streamlit application.
2. Upload a chest X-ray image in a supported image format.
3. View the predicted class.
4. Inspect the confidence score and probabilities for Normal and Pneumonia.
5. Review the Grad-CAM heatmap, when available.
6. Ask the conversational assistant questions about the output, if the chatbot is configured and available.
7. Interpret the result as an experimental model prediction, not a medical diagnosis.

## Current Limitations

- The classifier is trained for only two categories: Normal and Pneumonia.
- Test performance is lower than validation performance.
- The model produces false-positive predictions on normal images.
- External images may differ from the training dataset and produce unreliable results.
- Confidence scores are model probabilities, not measures of clinical certainty.
- Grad-CAM highlights influential regions but does not prove the presence or location of disease.
- Gemini functionality requires API configuration, network access, and available API quota.
- The system has not been established as a clinically validated medical device.

## Future Scope

- Investigate false-positive predictions and improve generalization.
- Evaluate robustness across different datasets and image sources.
- Add more comprehensive evaluation and error-analysis reports.
- Improve the conversational assistant's grounding in current prediction context.
- Strengthen input validation, exception handling, and user experience.
- Explore additional explainability and model-calibration methods.
- Conduct appropriate external validation before considering any clinical use.

## Team

Developed as an academic computer science project focused on deep learning, explainable AI, and interactive applications.

Add your team members' names, roles, and institutional details here.

## Disclaimer

GroundedCXR is intended strictly for educational and research purposes. It is not intended to diagnose, treat, cure, or prevent any disease. Do not make medical decisions based on its predictions or visualizations. Consult a qualified healthcare professional for medical evaluation.

## Acknowledgments

- Paul Mooney for the Chest X-Ray Images (Pneumonia) dataset.
- PyTorch and Torchvision for deep learning tools.
- The Grad-CAM project for explainability functionality.
- Streamlit for the interactive application framework.
- Google for the Gemini API, where configured.

## License

A license has not yet been specified for this repository. Add an appropriate project license before permitting others to reuse, modify, or distribute the source code.

The dataset has its own licensing terms; refer to the original Kaggle dataset page for details.
