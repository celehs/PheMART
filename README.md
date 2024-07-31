# PheMART
This repository hosts the official implementation of PheMART, a method that predicts the phenotypic effects of missense variants (MV) via deep contrastive learning.        
All the source data will be published on: https://doi.org/10.6084/m9.figshare.26036227 and https://doi.org/10.5281/zenodo.13138603.   
We provide visualization of high-confidence phenotypic predictions at: https://shiny.parse-health.org/PheMART/. The visualizations are, both by phenotypes, in which we visualize all MVs predicted to be implicated in the phenotypes, and by genes in which we provide the density of pathogenic MVs at different mutation positions for varied phenotypes.    



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
