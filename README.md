# CytoAutoCluster: A Semi-Supervised Deep Learning Framework for Cytometry Data Analysis

## Overview
CytoAutoCluster is an advanced deep learning framework designed for cytometry data analysis. It combines semi-supervised learning, autoencoder-based dimensionality reduction, and clustering techniques to overcome challenges like high dimensionality, noisy data, and missing labels. This project also provides an interactive Gradio-based user interface for visualizing and exploring clustering results.

---

## Features
- *Preprocessing*: Handles data cleaning, feature imputation, and normalization.
- *Dimensionality Reduction*: Employs PCA and t-SNE for effective data visualization.
- *Self-Supervised Learning*: Trains an autoencoder to handle incomplete data and extract robust latent features.
- *Clustering*: Uses k-Means and Hierarchical Clustering with evaluation metrics like AUROC,Accuracy.
- *Gradio Interface*: Offers real-time interaction for visualizing clustering results and adjusting parameters.

---
*License*

This project is licensed under the MIT License. See LICENSE for details.

*Acknowledgments*

Dataset: Levine32Dimensional dataset sourced from Kaggle.
Frameworks and Libraries: TensorFlow, Scikit-learn, Gradio, Matplotlib, Pandas, Jupiter Notebook.
