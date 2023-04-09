# DL-based tumor type classification
[a.a. 22/23] Gaetano Antonucci & Nicola Pagliara

Project developed as part of course "**Strumenti formali per la bioinformatica**" of the Computer Science
Department (**Università degli Studi di Salerno**)

The aim of this work is to replicate and improve the work cited in the **Reference** section below. 

# Setup
To create the environment used to perform the tests we used Anaconda (python 3.9) with the following packages:

|                 Package | Version                   |
|------------------------:|:--------------------------|
|                   numpy | 1.23.5                    |
|                  pandas | 1.5.2                     |
|                 pytorch | 1.13.1 CUDA (ver. 11.6)   |
|              matplotlib | 3.5.3                     |
|            scikit-learn | 1.2.1                     |
|            scikit-image | 0.19.3                    |
|        imbalanced-learn | 0.10.1                    |
|                  opencv | 4.7.0                     |

# Data
## Raw Data
Data was obtained from [GDAC FireHose](https://gdac.broadinstitute.org/) (managed by Broad Institute)

Just for simplicity we include the link supplied by B.Lyu and A. Haque from which we downloaded the raw data and the annotation file
needed in preprocessing:
https://drive.google.com/drive/folders/1LfOiyMgnoQy3jaJ37jLeARfw7riLwkyW

## Processed Data
Our elaboration of data is available at:
https://drive.google.com/drive/folders/1mn8Sis3rsQAAaAHoALNKTtFy3mwOoUYg

# Running
To run the application, please check the paths in every script and then run the scripts by numeric order.

NOTE: please, remind that ```raw_data.py``` was used only in binary and ternary test.
# Hardware
All tests was performed on a HP Server with motherboard **HPE ProLiant ML350 Gen 10** in this setup:

|                      |                                                                       |
|---------------------:|:----------------------------------------------------------------------|
| **Operative System** | Ubuntu 21.10 (64-bit)                                                 |
|              **CPU** | Intel(R) Xeon(R) Gold 5218 CPU @ 2.30Ghz (x32)                        |
|              **RAM** | 270 GB                                                                |
|              **GPU** | NVIDIA Quadro RTX 4000 (8 GiB of dedicated memory) \[CUDA ver. 11.6\] |
# Pipeline
Our application was structured in modules as follows:

![Project Pipeline](/deliverables/Pipeline.png)
# Filesystem Organization
Application's output files was organized as follows:

![Project Output files filesystem organization](/deliverables/Filesystem.png)

# Reference

This work is based on the work available at: https://dl.acm.org/citation.cfm?id=3233588

[GitHub repository](https://github.com/HHHit/DL-based-Tumor-Classification)

Boyu Lyu and Anamul Haque. 2018. Deep Learning Based Tumor Type Classification Using Gene Expression Data. In Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics (BCB '18). ACM, New York, NY, USA, 89-96. DOI: https://doi.org/10.1145/3233547.3233588
