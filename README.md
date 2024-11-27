# CytoAutoCluster: A Semi-Supervised Deep Learning Framework for Cytometry Data Analysis

## Overview
CytoAutoCluster is a semi-supervised deep learning framework designed to simplify and improve cytometry data analysis. By leveraging advanced techniques such as autoencoder-based dimensionality reduction and clustering, this project handles challenges like high-dimensional, noisy, and partially labeled data. It integrates modern data preprocessing, clustering, and visualization methods, along with a user-friendly Gradio interface.

## Features
- **Data Cleaning & Preprocessing**: Handles missing data, imputes values, and normalizes features.
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA)
  - t-SNE for visualization
- **Self-Supervised Learning**: An autoencoder reconstructs data with simulated missing values.
- **Clustering Algorithms**: k-Means and Hierarchical Clustering.
- **Performance Metrics**: Adjusted Rand Index, Silhouette Score, and Reconstruction Loss.
- **Gradio Interface**: Real-time clustering visualization and user interaction.

## Dataset
- **Name**: Levine32Dimensional (from Kaggle)
- **Properties**:
  - 265,627 cells and 32 markers.
  - 61% unlabeled data.
  - 14 manually gated clusters.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/CytoAutoCluster.git
   cd CytoAutoCluster
