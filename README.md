# CytoAutoCluster: A Semi-Supervised Deep Learning Framework for Cytometry Data Analysis

## Overview
CytoAutoCluster is an advanced deep learning framework designed to tackle challenges in cytometry data analysis, including high dimensionality, missing labels, and noise. It leverages semi-supervised learning, dimensionality reduction, and clustering techniques with a focus on interactive data exploration using Gradio.

---

## Repository Structure

```plaintext
main/
├── Mohana/                        # Supporting folder for custom scripts or files
├── CytoAutoClusterCode.docx       # Detailed documentation and project notes
├── ENCODER_AND_GRADIO.ipynb       # Jupyter notebook combining encoder training and Gradio interface
├── README.md                      # Project documentation (this file)
├── cytoAutoCluster_BeforeEncoder.ipynb  # Notebook for initial data exploration and preprocessing
├── LICENSE                        # Licensing information for the project


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
   cd CytoAutoClusterom/MOHANAL/CytoAutoCluster.git
   cd CytoAutoCluster
2.Install required dependencies:
  ```bash
  pip install -r requirements.txt



