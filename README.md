# FakeNewsShield

---
A multimodal AI system that detects misinformation in social media posts by combining Large Language Models and Vision Transformers for enhanced accuracy.

---

## Project Overview

**FakeNewsShield** is an intelligent system that detects misinformation by analyzing both text and images together (multimodal approach). Unlike traditional systems that analyze only text or only images, this system uses fusion techniques to understand the relationship between caption and visual content.

---

## Objectives

Develop a multimodal misinformation detection system
Provide explainable AI through attention heatmaps
Create a user-friendly web interface
Generate downloadable analysis reports.

---

## Project Youtube Demonstration

Video:Watch the full demo here  

[YouTube Demo](https://youtu.be/GMQZ6kBX_x0?si=n9qlMrN-nIeQlbga)

---

## Technologies Used

**Deep Learning Models**: RoBERTa (Text) + Vision Transformer (ViT) (Image)

**Framework**: PyTorch + Hugging Face Transformers

**Interface**: Streamlit

**Visualization**: Matplotlib (Attention Heatmap)

**Database**: SQLite

**Other**: PIL, NumPy

---

## Virtual Environment (venv) Setup Guide

A virtual environment (venv) in Python is an isolated workspace that allows you to install and manage project-specific dependencies separately from your system-wide Python installation.

### Clone the Repository and venv setup

```powershell
git clone https://github.com/feyap88/FakeNewsShield.git
cd FakeNewsShield

# Create virtual environment
python -m venv venv

# On Windows (PowerShell)
venv\Scripts\activate

# You should see (venv) at the beginning of your terminal prompt

#### Install Required Packages

pip install -r requirements.txt

then run
streamlit run app.py

```

## Model Setup

Extracting the Models Folder

The project includes a compressed file containing pre-trained models and supporting files.

Locate the zipped folder

You will find a file such as:
models.zip
Unzip the folder

### Extract the models folder

```powershell
Expand-Archive -Path models.zip -DestinationPath . -Force

```

## Training the Model

Activate virtual environment
**venv\Scripts\activate**

Run training (Best done on Google Colab with GPU),
recommended to use Google Colab (GPU enabled) for faster training, especially for large datasets.
**python -m src.train**

Ensure all required dependencies are installed before training

## How the System Works

### Step-by-Step Process

1. **Input Collection**
   - User pastes the post text/caption
   - User uploads an accompanying image
   - Optional: Selects the platform/source (WhatsApp, X, Facebook, etc.)

2. **Feature Extraction**
   - **Text Processing**: The text is passed through **RoBERTa** (a powerful language model) to extract semantic meaning and context.
   - **Image Processing**: The image is passed through **Vision Transformer (ViT)** to extract visual features.

3. **Multimodal Fusion**
   - Features from both text and image are combined using **Cross-Attention** mechanism.
   - This allows the model to understand the relationship and consistency between the text and the image.

4. **Classification**
   - The fused features are passed to a classification layer.
   - The model outputs two probabilities: **REAL** or **FAKE**.

5. **Explainable AI**
   - An **Attention Heatmap** is generated to show which parts of the image contributed most to the decision.
   - A natural language explanation is provided to help users understand why the content was classified as real or fake.

6. **Output & Reporting**
   - Displays prediction with confidence score
   - Shows detailed AI explanation
   - Generates downloadable analysis report (TXT)
   - Saves the analysis in the history database

---
