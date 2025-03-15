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
- **Storage**: At least 10GB of free space 

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
- **Required Input**: A list of missense variants in **CSV** format and a file containing the variant embeddings with ***Numpy array** format. Each row represents the embedding vector of a variant.
- **Example format** (CSV):
  ```
  variants
  NM_002074.5(GNB1):c.230G>A (p.Gly77Asp)
  NM_022787.4(NMNAT1):c.205A>G (p.Met69Val)
  ```

### 2. Fine-Tuning or Training a New Model
- **Variant Embeddings**: Precomputed embeddings for missense variants.
- **Phenotype Embeddings**: Precomputed phenotype representation. The LLM phenotype embeddings have been provided and the EHR embedding will be provided upon request.
- **Training Labels (if available)**: Variant-phenotype annotations in **CSV** format.
  - **Example format** (CSV):
  ```
  variant_id,gene,phenotype_CUI
  NM_002074.5(GNB1):c.388G>A (p.Glu130Lys),GeneX,C3276355
  NM_003036.4(SKI):c.68A>C (p.Gln23Pro),GeneY,C1321551
  ```

**Note:** We also provide scripts to preprocess user-provided patient-level data for generating EHR embeddings.

---

## Usage
### Running Inference with Pre-trained Model
To run inference using our pre-trained model:
```bash
python predict.py --file_snp_prediction variants.csv  --dirr_results_main  result/ --pretrained_model data/model_pretrained/
```

#### Arguments
- `--file_snp_prediction`: File containing the list of variants to predict (CSV).
- `--dirr_results_main`: Path to save the predictions.
- `--pretrained_model`: Path to the pre-trained model.

---

### Fine-Tuning or Training a New Model
To fine-tune or train a model using your own dataset:
```bash
bash submit.sh --train --file_annotations /path/to/annotations --file_snps_labeled  /path/to/list of labeled variants  --file_snps_labeled_embedding  /path/to/embeddings of labeled variants  --dirr_results_main /path/to/results  --dirr_save_model  /path/to/saved model
```

#### Arguments
- `--train`: Flag to indicate training mode.
- `--file_annotations`: File containing the annotated variant-phenotype pairs.
- `--file_annotations`: File containing the list of annotated variants.
- `--file_annotations`: File containing the embedding vectors of the annotated variants.
- `--dirr_results_main`: Path to save result files.
- `--dirr_save_model`: Path to save the trained model.

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
- `training_logs.txt`: Training loss, accuracy, and hyperparameters.
- `predictions_on_validation.csv`: Model performance on the validation dataset.

---

## Computation Steps
The `submit.sh` script automates the following computational steps:

1. **Data Preprocessing**
   - Converts CSV variant data into embeddings.
   - Get phenotype embeddings.
   - Splits data into training and validation sets.

2. **Model Training / Fine-Tuning**
   - Loads the dataset and initializes the neural network.
   - Runs training with mini-batch gradient descent.
   - Performs validation and saves the models.

3. **Prediction and Evaluation**
   - If in inference mode, loads the trained model and predicts pathogenicity scores.
   - Saves predictions in `results.csv`.

---
