# PheMART: Phenotypic Prediction of Missense Variants via Deep Contrastive Learning

**This repository hosts the official implementation of PheMART, a method that predicts the phenotypic effects of missense variants (MV) via deep contrastive learning.        
All the source data will be published on: https://doi.org/10.6084/m9.figshare.26036227 and https://doi.org/10.5281/zenodo.13138603.   
We offer visualizations of high-confidence phenotypic predictions at: https://shiny.parse-health.org/PheMART/. These visualizations are categorized by phenotypes, genes and protein domains.**  

---

## Table of Contents
- [Introduction](#introduction)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Input Data](#input-data)
- [Usage](#usage)
  - [Running Inference with Pre-trained Model](#running-inference-with-pre-trained-model)
  - [Fine-Tuning or Training a New Model](#fine-tuning-or-training-a-new-model)
- [Expected Outputs](#expected-outputs)
- [Computation Steps](#computation-steps)
- [Citation](#citation)
- [License](#license)

---

## Introduction
PheMART is a computational framework designed for **predict the clinical phenotypic effects of missense variants**. Users can either:
- **Use a pre-trained model** for quick inference.
- **Fine-tune or train the model** using their own dataset.

This README provides step-by-step instructions for setting up, running, and interpreting the results.

---

## System Requirements
To ensure smooth execution, we recommend the following system specifications:

- **Operating System**: Ubuntu 20.04 / macOS 12+ / Windows Subsystem for Linux (WSL)
- **Python Version**: 3.8+
- **GPU (for training)**: NVIDIA GPU with at least 16GB VRAM (e.g., RTX 3090, A100, V100)
- **CUDA**: 10.4+
- **Memory**: Minimum 16GB RAM (32GB recommended for large datasets)
- **Storage**: At least 10GB free space 

---

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/celehs/PheMART.git
cd PheMART
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python3 -m venv pheMART_env
source pheMART_env/bin/activate  # On Windows: pheMART_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Input Data
PheMART requires different input datasets depending on the use case.

### 1. Using the Pre-trained Model
- **Required Input**: A list of missense variants in **CSV** format.
- **Example format** (CSV):
  ```
  variant_id,gene,chromosome,position,ref,alt
  rs123456,GeneX,1,123456,A,T
  rs789012,GeneY,2,789012,C,G
  ```

### 2. Fine-Tuning or Training a New Model
- **Variant Embeddings**: Precomputed embeddings for missense variants.
- **Phenotype Embeddings**: Precomputed phenotype representation. The LLM phenotype embeddings have been provided and the EHR embedding will be provided upon request.
- **Training Labels (if available)**: Variant-pathogenicity annotations.

**Note:** We also provide scripts to preprocess user-provided patient-level data for generating EHR embeddings..

---

## Usage
### Running Inference with Pre-trained Model
To run inference using our pre-trained model:
```bash
python predict.py --file_snp_prediction variants.csv  --dirr_results_main  results.csv --pretrained_model data/model_pretrained/
```

#### Arguments
- `--input`: Path to the input variant file (VCF/CSV).
- `--output`: Path to save the predictions.
- `--pretrained_model`: Path to the pre-trained model.

---

### Fine-Tuning or Training a New Model
To fine-tune or train a model using your own dataset:
```bash
bash submit.sh --train --data_path /path/to/dataset --output_dir /path/to/output
```

#### Arguments
- `--train`: Flag to indicate training mode.
- `--data_path`: Directory containing input variant & phenotype embeddings.
- `--output_dir`: Path to save the trained model and logs.

---

## Expected Outputs
PheMART generates different outputs depending on the mode of operation.

### 1. Inference Mode (Using Pre-trained Model)
- `results.csv`: Contains predicted pathogenicity scores for each variant.
  ```
  variant_id,predicted_score
  rs123456,0.85
  rs789012,0.15
  ```
  - Higher scores indicate a higher likelihood of pathogenicity.

### 2. Training Mode
- `model_checkpoint.pth`: The saved fine-tuned model.
- `training_logs.txt`: Training loss, accuracy, and hyperparameters.
- `predictions_on_validation.csv`: Model performance on the validation dataset.

---

## Computation Steps
The `submit.sh` script automates the following computational steps:

1. **Data Preprocessing**
   - Converts VCF/CSV variant data into embeddings.
   - Normalizes phenotype embeddings.
   - Splits data into training and validation sets.

2. **Model Training / Fine-Tuning**
   - Loads the dataset and initializes the neural network.
   - Runs training with mini-batch gradient descent.
   - Performs validation and saves the best-performing model.

3. **Prediction and Evaluation**
   - If in inference mode, loads the trained model and predicts pathogenicity scores.
   - Saves predictions in `results.csv`.

---
