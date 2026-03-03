# Deep-learning-image-Classifier

## Overview
This repository features an end-to-end Deep Learning solution for high-accuracy dog breed identification. By utilizing **Transfer Learning** with the **MobileNet V2** architecture, the system can classify 120 distinct breeds. The project is built with a focus on "Production-Ready" code, moving beyond simple notebooks into a structured, automated inference pipeline.

## Key Features
* **Transfer Learning Optimization:** Leverages pre-trained weights from MobileNet V2 (via TensorFlow Hub) to achieve high-performance classification with optimized training efficiency.
* **Comprehensive Research & Development:** Includes a detailed Jupyter Notebook covering the entire lifecycle-from raw data exploration (EDA) and preprocessing to model architecture design and training from scratch.
* **Automated Inference Pipeline:** A dedicated source script that manages the production flow, including image normalization, tensor transformation, and batch prediction.
* **Visual Discovery Gallery:** Automatically generates a categorized Matplotlib grid summary at the end of each run, providing an intuitive visual report of all classification results.



## Project Structure
```text
├── models/                # Serialized model weights (.h5) and class labels (.json)
├── notebooks/             # Research phase: EDA and Model Training
├── src/                   # Production-grade source code
│   └── prediction-pipline.py  # Main execution entry point
├── images_to_predict/     # Input directory for inference
├── venv/                  # Virtual environment (Not uploaded to Git)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup & Installation

### 1. Prerequisites
* **Python 3.10+** (Required for TensorFlow 2.15 compatibility)

### 2. Clone the Repository
```bash
git clone https://github.com/lotantamary/Deep-learning-image-Classifier.git
cd Deep-learning-image-Classifier
```
### 3. Environment Setup
It is highly recommended to use a virtual environment to isolate dependencies:
```bash
# Create the virtual environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 4. Install Dependencies
The `requirements.txt` file contains all the necessary libraries to run the research notebooks and the production pipeline, including **TensorFlow**, **TF-Keras**, and **Matplotlib**:

```bash
# Ensure your virtual environment is activated before running this
pip install -r requirements.txt
```

## Usage & Execution

### 1. Research & Model Development

To explore the data analysis, feature engineering, and the model training process, you can launch the interactive notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lotantamary/Deep-learning-image-Classifier/blob/main/notebooks/Dog_Breeds_Deep_Learning_Classifier_Notebook.ipynb)

*The notebook includes Data Exploration (EDA), model architecture, and performance evaluation.*

### 2. Prepare Input Images
Place the dog images you wish to classify inside the `images_to_predict/` directory. The pipeline supports **.jpg** and **.jpeg** formats.  
Note: A sample image is already included in the images_to_predict/ directory for an immediate demonstration of the pipeline.

### 3. Run the Prediction Pipeline
From the project root directory, execute the following command:

```bash
python src/prediction-pipline.py
```

### 3. Pipeline Workflow
The execution script is engineered to handle the entire data-to-prediction lifecycle through a series of "Production-Grade" stages:

1.  **Environment Validation:** The system performs a startup check to ensure all required assets (Model weights `.h5` and Class labels `.json`) are present and readable.
2.  **Image Preprocessing:** * **Loading:** Reads raw image data from the input directory.
    * **Normalization:** Converts pixel values to the `[0, 1]` range required by the neural network.
    * **Resizing:** Scales all input images to a uniform 224x224 resolution.
3.  **Batch Inference:** Efficiently processes the image tensors through the **MobileNet V2** architecture to generate probability distributions for 120 breeds.
4.  **Console Reporting:** Generates a structured, formatted table in the terminal displaying filenames, predicted breeds, and confidence percentages.
5.  **Visual Results Gallery:** Automatically triggers a **Matplotlib-based UI** that displays a visual summary, allowing for immediate verification of the model's accuracy.