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
- **Size**: 265,627 rows × 41 features
- **Labels**: 39% labeled, 61% unlabeled
- **Challenges**:
  - High dimensionality of 32 Markers.
  - Large proportion of missing labels.

---

Repository Structure
        ```plaintext

            main/
            ├── Mohana/                        # Supporting folder for custom scripts or files
            ├── CytoAutoClusterCode.pdf       # Detailed documentation and project notes
            ├── README.md                      # Project documentation (this file)
            ├──CytoAutoClusterCode/
                ├── cytoAutoCluster_BeforeEncoder.ipynb  # Notebook for initial data exploration and preprocessing
                ├── ENCODER_AND_GRADIO.ipynb       # Jupyter notebook combining encoder training and Gradio interface
            ├── LICENSE                        # Licensing information for the project
---
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/MOHANAL/CytoAutoCluster.git
   cd CytoAutoCluster
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
---
## Usage
## 1. Preprocessing
  Run the **cytoAutoCluster_BeforeEncoder.ipynb** notebook for initial data cleaning and standardization.

## 2. Train Autoencoder and Clustering Models
  Use the **ENCODER_AND_GRADIO.ipynb** notebook to:
  Train the autoencoder.
  Perform clustering on the dataset.

## 3. Launch Gradio Interface
   Execute the relevant cells in the **ENCODER_AND_GRADIO.ipynb** notebook to launch the Gradio interface for real-time data exploration and clustering visualization.
  
---
## Results
**Key Metrics:**

**Adjusted Rand Index (ARI): **~0.85****
  
  Indicates strong alignment with the ground truth labels.
  
**Silhouette Score: **0.74****
  
  Demonstrates well-separated and compact clusters.
  
**Reconstruction Loss (MSE):**

  Low reconstruction loss confirms the autoencoder's effectiveness.
  
**Accuracy:0.932**
  
## Visual Insights:
## PCA:
2D and 3D plots reveal distinct clustering patterns.
## t-SNE:
Highlights natural grouping tendencies in low-dimensional space.

---
**Contribution**

We welcome contributions! Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes and open a pull request.
---
**License**

This project is licensed under the MIT License. See LICENSE for details.

**Acknowledgments**

Dataset: Levine32Dimensional dataset sourced from Kaggle.
Frameworks and Libraries: TensorFlow, Scikit-learn, Gradio, Matplotlib, Pandas, NumPy.

