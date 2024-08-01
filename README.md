# PheMART
This repository hosts the official implementation of PheMART, a method that predicts the phenotypic effects of missense variants (MV) via deep contrastive learning.        
All the source data will be published on: https://doi.org/10.6084/m9.figshare.26036227 and https://doi.org/10.5281/zenodo.13138603.   
We offer visualizations of high-confidence phenotypic predictions at: https://shiny.parse-health.org/PheMART/. These visualizations are categorized both by phenotypes, showcasing all MVs predicted to be implicated in each phenotype, and by genes, displaying the density of pathogenic MVs at different mutation positions across various phenotypes.   



# Requirements
* python 3.7
* tensorflow==2.5.2
* numpy >= 1.19
* pandas >= 1.3
* scikit-learn >= 1.0.2

# Usage
```sh
  1. first install the environment:  PheMART_setup.sh
  2. PheMART training and evaluation: PheMART_submit.sh
```
