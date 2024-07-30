# PheMART
This repository hosts the official implementation of PheMART, a method that can predict the phenotypic effects of missense variants via deep contrastive learning.        
All the source data will be published on: https://doi.org/10.6084/m9.figshare.26036227   
We provide high-confidence phenotypic predictions at: https://shiny.parse-health.org/PheMART/. We provide visualizations, both by phenotypes, in which we provide all MVs predicted to be implicated in the phenotypes, and by genes in which we provide the density of pathogenic variants at different mutation positions for varied phenotypes.    



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
