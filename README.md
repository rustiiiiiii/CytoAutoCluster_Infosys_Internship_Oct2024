# CytoAutoCluster: A Semi-Supervised Deep Learning Framework for Cytometry Data Analysis

## Overview
CytoAutoCluster is an advanced deep learning framework designed for cytometry data analysis. It combines semi-supervised learning, autoencoder-based dimensionality reduction, and clustering techniques to overcome challenges like high dimensionality, noisy data, and missing labels. This project also provides an interactive Gradio-based user interface for visualizing and exploring clustering results.

---

## Features
- **Preprocessing**: Handles data cleaning, feature imputation, and normalization.
- **Dimensionality Reduction**: Employs PCA and t-SNE for effective data visualization.
- **Self-Supervised Learning**: Trains an autoencoder to handle incomplete data and extract robust latent features.
- **Clustering**: Uses k-Means and Hierarchical Clustering with evaluation metrics like ARI and Silhouette Score.
- **Gradio Interface**: Offers real-time interaction for visualizing clustering results and adjusting parameters.

---

## Dataset Information
- **Dataset**: Levine32Dimensional (sourced from Kaggle)
- **Size**: 265,627 rows × 32 features
- **Labels**: 39% labeled, 61% unlabeled
- **Challenges**:
  - High dimensionality of 32 features.
  - Large proportion of missing labels.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/CytoAutoCluster.git
   cd CytoAutoCluster
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
# 1. Preprocessing
  Run the **cytoAutoCluster_BeforeEncoder.ipynb** notebook for initial data cleaning and standardization.

# 2. Train Autoencoder and Clustering Models
  Use the **ENCODER_AND_GRADIO.ipynb** notebook to:
  Train the autoencoder.
  Perform clustering on the dataset.

# 3. Launch Gradio Interface
Execute the relevant cells in the **ENCODER_AND_GRADIO.ipynb** notebook to launch the Gradio interface for real-time data exploration and clustering visualization.
